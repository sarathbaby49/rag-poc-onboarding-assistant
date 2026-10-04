"""Layer 5 — Token economics: what did that call cost?  (EXERCISE C1)

Every model call is billed in tokens. The gateway tells us how many tokens a
call used; this module turns that into dollars.

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
