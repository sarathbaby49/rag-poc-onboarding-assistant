"""Layer 5 — Model selection & cost: all four exercises live here (C1–C4).

Every model call is billed in tokens. The gateway tells us how many tokens a
call used; C1 turns that into dollars, C2/C3 decide which model tier handles
a question or graph step, and C4 builds a prompt that lets the provider cache
the unchanging part of it.

    cost = input tokens  x input price
         + output tokens x output price
         + cached tokens x (input price x a cache multiplier)

Prices are USD per 1 MILLION tokens (standard tier), checked 4 Oct 2026 on the
provider pricing pages. They change often — re-check before you rely on them:
  https://platform.claude.com/docs/en/about-claude/pricing
  https://developers.openai.com/api/docs/pricing
  https://ai.google.dev/gemini-api/docs/pricing

`usage` is always this normalized dict (src/cost_helper.py builds it for you
from whatever the gateway returns):

    {
        "input_tokens":       fresh (uncached) input tokens, full price
        "output_tokens":      generated tokens, output price
        "cache_read_tokens":  input tokens served from the prompt cache
        "cache_write_tokens": input tokens written into the prompt cache
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from src import config

CHEAP_MODEL = config.CHEAP_MODEL
MID_MODEL = config.MID_MODEL
STRONG_MODEL = config.STRONG_MODEL


# ============================================================================
# C1 — Cost meter
# ============================================================================


@dataclass(frozen=True)
class Price:
    input: float              # USD per 1M input tokens
    output: float             # USD per 1M output tokens
    cache_read: float = 0.1   # multiplier on `input` for a cache hit
    cache_write: float = 1.0  # multiplier on `input` for writing the cache


# Keys are matched against the END of the model name, so
# "litellm_proxy/anthropic/claude-sonnet-5-5" finds "claude-sonnet-5-5".
PRICES: dict[str, Price] = {
    # Anthropic — 5-minute cache writes cost 1.25x; reads 0.1x (newer models less)
    "claude-fable-5-1":   Price(10.00, 50.00, cache_read=0.025, cache_write=1.25),
    "claude-opus-5-5":    Price(4.00, 20.00, cache_read=0.05, cache_write=1.25),
    "claude-sonnet-5-5":  Price(2.00, 10.00, cache_read=0.10, cache_write=1.25),
    "claude-haiku-4-5":   Price(1.00, 5.00, cache_read=0.10, cache_write=1.25),
    # OpenAI — caching is automatic; cached input is listed at 5-10% of input
    "gpt-6-astra":        Price(10.00, 50.00, cache_read=0.10),
    "gpt-6.1-sol":        Price(2.00, 10.00, cache_read=0.05),
    "gpt-6-luna":         Price(0.10, 0.50, cache_read=0.10),
    "gpt-4.1-mini":       Price(0.40, 1.60, cache_read=0.25),  # legacy lab default
    # Google — context caching also charges hourly storage (not modelled here)
    "gemini-3.1-pro-preview": Price(2.00, 12.00, cache_read=0.10),
    "gemini-3.8-flash":       Price(0.75, 3.75, cache_read=0.10),  # promo until 31 Dec 2026
    "gemini-3.5-flash-lite":  Price(0.30, 2.50, cache_read=0.10),
}

PER_MILLION = 1_000_000


def price_for(model: str) -> Price:
    """Find the price entry for a model name (with or without a gateway prefix)."""
    name = model.lower()
    for key, price in PRICES.items():
        if name.endswith(key):
            return price
    raise KeyError(
        f"No price for model {model!r}. Add it to PRICES in src/cost.py "
        "(check the provider's pricing page)."
    )


def cost_of(usage: dict, model: str) -> float:
    """EXERCISE C1 — return the USD cost of one call.

    Steps:
      1. price = price_for(model)
      2. fresh input:   usage["input_tokens"]       x price.input
      3. output:        usage["output_tokens"]      x price.output
      4. cache reads:   usage["cache_read_tokens"]  x price.input x price.cache_read
      5. cache writes:  usage["cache_write_tokens"] x price.input x price.cache_write
      6. add them up and divide by PER_MILLION (prices are per 1M tokens)

    Treat a missing key as 0 (use usage.get(..., 0)).

    Sanity check: 10,000 fresh input tokens on Sonnet 5.5 ($2 / 1M) = $0.02.
    """
    raise NotImplementedError("Exercise C1: implement cost_of — see layer5 EXERCISES.md")


def monthly_cost(cost_per_question: float, questions_per_month: int = 100_000) -> float:
    """Scale one question's cost to a month (done for you)."""
    return cost_per_question * questions_per_month


# ============================================================================
# C2 + C3 — Model selection & routing
#
# Not every question needs the most expensive model. "What's the repo URL?"
# can go to a cheap, fast model; "Why is my migration failing?" needs a strong
# one. Routing is the single biggest cost lever: on our numbers, sending 60%
# of questions to the cheap tier cuts the monthly bill by more than half.
#
# C2 — pick_model(question): route one user question to a tier.
# C3 — model_for_step(step, attempts): pick a tier per step of the LangGraph
#      onboarding-plan bot ("Create a 10-day onboarding plan for a new
#      backend engineer"). The Cost Lab wires your function into the graph
#      for you.
# ============================================================================


def pick_model(question: str) -> str:
    """EXERCISE C2 — return the model to use for this question.

    Right now everything goes to STRONG_MODEL (safe, but the most expensive).
    Replace it with a simple rule, for example:
      - debugging / reasoning words ("why", "error", "fail", "fix", "explain",
        "compare", "debug"), stack traces or code     -> STRONG_MODEL
      - very long questions (say > 150 characters)     -> STRONG_MODEL
      - short factual lookups ("what's the repo URL?") -> CHEAP_MODEL

    Self-check: python -m checks.check_routing  (12 labelled questions, need 10+)
    Bonus: instead of rules, ask CHEAP_MODEL to label the question "easy" or
    "hard" and route on its answer. Is the extra call worth it?
    """
    return STRONG_MODEL


# The three steps of the plan bot that call a model:
#   "parse_request" — pull the role and number of days out of the request (extraction)
#   "draft_plan"    — write the day-by-day plan (needs judgment)
#   "replan"        — fix ungrounded steps or apply mentor feedback
STEPS = ("parse_request", "draft_plan", "replan")


def model_for_step(step: str, attempts: int = 0) -> str:
    """EXERCISE C3 — return the model for one step of the plan graph.

    `attempts` is how many replans have already happened (0 on the first replan).

    Target:
      - "parse_request" -> CHEAP_MODEL   (simple extraction)
      - "draft_plan"    -> MID_MODEL     (writing needs judgment)
      - "replan"        -> MID_MODEL, but escalate to STRONG_MODEL once
                           attempts >= 2 (cheaper tries already failed)

    The other nodes (retrieve, check_grounding, publish) never call a model,
    and mentor_approval is a person. Self-check: python -m checks.check_routing
    """
    return STRONG_MODEL


# ============================================================================
# C4 — Prompt caching: static first, dynamic last
#
# The provider can cache the processed START of a prompt. If the next request
# begins with exactly the same text, those tokens are billed at ~10% of the
# input price (Anthropic: write 1.25x once, then read 0.1x — see PRICES above).
#
# The cache matches from the very first token, so ONE changed character early
# in the prompt (a timestamp, the user's name, the question) breaks it for
# everything after. Rule: put what never changes first, what changes every
# call last.
#
# Here the static part is the "team handbook": every doc and code file in
# data/sample_company (a few thousand tokens). Small corpora like ours can
# simply ship the whole handbook in a cached prefix.
# ============================================================================

SYSTEM_PROMPT = (
    "You are the Acme Shop onboarding assistant. Answer new engineers' questions "
    "using ONLY the team handbook and any extra context below. Cite the file names "
    "you used. If the answer isn't there, say you don't know."
)


@lru_cache(maxsize=1)
def handbook() -> str:
    """Every doc + code file, in a fixed order (done for you).

    Sorted so the text is byte-identical on every call — a requirement for caching.
    """
    root = config.DATA_DIR
    files = sorted(root.glob("*.md")) + sorted((root / "code").glob("*.py"))
    parts = [f"=== {p.relative_to(root)} ===\n{p.read_text()}" for p in files]
    return "\n\n".join(parts)


def format_hits(hits: list[dict] | None) -> str:
    """Retrieved chunks as a numbered block (done for you)."""
    if not hits:
        return "(none)"
    return "\n\n".join(f"[{i}] ({h['source']}) {h['text']}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[dict] | None = None) -> list[dict]:
    """EXERCISE C4 — return cache-friendly messages for the gateway.

    This version WORKS but defeats caching three ways:
      1. a timestamp at the very top changes the prefix on every call
      2. the question comes before the handbook, so the prefix differs per question
      3. nothing is marked for caching

    Rewrite it so that:
      - messages[0] is a system message whose content is a list with ONE text
        block: SYSTEM_PROMPT + "\\n\\n" + handbook(), carrying
        "cache_control": {"type": "ephemeral"}
          [{"role": "system", "content": [{"type": "text", "text": ...,
                                           "cache_control": {"type": "ephemeral"}}]}]
      - the LAST message is the user message with the dynamic parts:
        format_hits(hits) and then the question
      - no timestamp anywhere in the static part

    Self-check: python -m checks.check_caching  (live part needs the gateway)
    """
    now = datetime.now().isoformat(timespec="seconds")
    return [
        {
            "role": "system",
            "content": (
                f"Current time: {now}\n"
                f"Question: {question}\n\n"
                f"{SYSTEM_PROMPT}\n\nTeam handbook:\n{handbook()}\n\n"
                f"Extra context:\n{format_hits(hits)}"
            ),
        },
        {"role": "user", "content": question},
    ]
