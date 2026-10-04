"""Self-check for Layer 5 exercise C1 (cost_of). Pure math — NO key needed.

Run from the repo root:  python -m checks.check_cost
"""

from __future__ import annotations

import sys

from src.cost import cost_of

SONNET = "litellm_proxy/anthropic/claude-sonnet-5-5"
OPUS = "litellm_proxy/anthropic/claude-opus-5-5"


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def main() -> int:
    print("C1 · cost_of\n")
    try:
        cost_of({"input_tokens": 1}, SONNET)
    except NotImplementedError:
        print("❌ C1 not started — implement cost_of in src/cost.py")
        return 1

    results = [
        _check("10K fresh input on Sonnet 5.5 = $0.020",
               _close(cost_of({"input_tokens": 10_000}, SONNET), 0.02)),
        _check("1K output on Sonnet 5.5 = $0.010",
               _close(cost_of({"output_tokens": 1_000}, SONNET), 0.01)),
        _check("10K cache WRITE on Sonnet 5.5 = $0.025 (1.25x)",
               _close(cost_of({"cache_write_tokens": 10_000}, SONNET), 0.025)),
        _check("10K cache READ on Sonnet 5.5 = $0.002 (0.1x)",
               _close(cost_of({"cache_read_tokens": 10_000}, SONNET), 0.002)),
        _check("10K cache READ on Opus 5.5 = $0.002 (0.05x of $4)",
               _close(cost_of({"cache_read_tokens": 10_000}, OPUS), 0.002)),
        _check("Missing keys count as zero",
               _close(cost_of({}, SONNET), 0.0)),
    ]

    # The deck's agent question: ~15K input + ~1K output on Opus 5.5 = $0.08
    per_q = cost_of({"input_tokens": 15_000, "output_tokens": 1_000}, OPUS)
    results.append(_check("Deck example: 15K in + 1K out on Opus 5.5 = $0.08", _close(per_q, 0.08)))

    # The caching slide: 10 calls, 10K-token prefix — 1 write + 9 reads
    cached = cost_of({"cache_write_tokens": 10_000}, SONNET) + 9 * cost_of({"cache_read_tokens": 10_000}, SONNET)
    results.append(_check("Caching slide: 1 write + 9 reads = $0.043", _close(cached, 0.043)))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
