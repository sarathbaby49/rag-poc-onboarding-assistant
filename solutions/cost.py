"""REFERENCE SOLUTION — Layer 5 — Model selection & cost (C1–C4).

Copy the function bodies into src/cost.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    """EXERCISE C1 — return the USD cost of one call."""
    price = price_for(model)
    total = (
        usage.get("input_tokens", 0) * price.input
        + usage.get("output_tokens", 0) * price.output
        + usage.get("cache_read_tokens", 0) * price.input * price.cache_read
        + usage.get("cache_write_tokens", 0) * price.input * price.cache_write
    )
    return total / PER_MILLION


def monthly_cost(cost_per_question: float, questions_per_month: int = 100_000) -> float:
    """Scale one question's cost to a month (done for you)."""
    return cost_per_question * questions_per_month


# ============================================================================
# C2 + C3 — Model selection & routing
# ============================================================================

# Words that signal reasoning, debugging or design work.
_HARD_WORDS = (
    "why", "error", "fail", "fix", "debug", "explain", "compare", "design",
    "trace", "exception", "broken", "crash", "refactor", "suggest", "trade-off",
)
# Code-ish signals: stack traces, file:line, HTTP error codes, snippets.
_CODE_PATTERN = re.compile(r"(traceback|\.py\b|line \d+|\b[45]\d\d\b|`|\(\)|::|=>)", re.I)


def pick_model(question: str) -> str:
    """EXERCISE C2 — return the model to use for this question."""
    q = question.lower()
    if len(q) > 150:
        return STRONG_MODEL
    if any(re.search(rf"\b{re.escape(w)}", q) for w in _HARD_WORDS):
        return STRONG_MODEL
    if _CODE_PATTERN.search(question):
        return STRONG_MODEL
    return CHEAP_MODEL


# The three steps of the plan bot that call a model:
#   "parse_request" — pull the role and number of days out of the request (extraction)
#   "draft_plan"    — write the day-by-day plan (needs judgment)
#   "replan"        — fix ungrounded steps or apply mentor feedback
STEPS = ("parse_request", "draft_plan", "replan")


def model_for_step(step: str, attempts: int = 0) -> str:
    """EXERCISE C3 — return the model for one step of the plan graph."""
    if step == "parse_request":
        return CHEAP_MODEL
    if step == "replan" and attempts >= 2:
        return STRONG_MODEL
    return MID_MODEL


# ============================================================================
# C4 — Prompt caching: static first, dynamic last
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
    """EXERCISE C4 — static, cached prefix first; dynamic context and question last."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT}\n\n{handbook()}",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": f"Extra context:\n{format_hits(hits)}\n\nQuestion: {question}",
        },
    ]
