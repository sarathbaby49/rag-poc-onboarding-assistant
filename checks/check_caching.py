"""Self-check for Layer 5 exercise C4 (cache-friendly prompt).

The structure checks need NO key. The live check (send the same prompt twice and
look for cache-read tokens) runs only when the LiteLLM gateway is configured.

Run from the repo root:  python -m checks.check_caching
"""

from __future__ import annotations

import sys
import time

from src import config
from src.cost import SYSTEM_PROMPT, build_messages, handbook

HITS = [{"text": "Run `make dev` to start the API on port 8000.", "source": "setup.md"}]


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def _first_text(msg: dict) -> str:
    content = msg["content"]
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content)
    return content


def _has_cache_control(msg: dict) -> bool:
    content = msg["content"]
    return isinstance(content, list) and any("cache_control" in block for block in content)


def structure_checks() -> bool:
    print("C4 · build_messages (structure)\n")
    a = build_messages("What's the repo URL?", HITS)
    time.sleep(1.1)  # a timestamp in the prefix would now differ
    b = build_messages("Why does my migration fail?", HITS)
    first_a, first_b = _first_text(a[0]), _first_text(b[0])
    results = [
        _check("First message is identical for two different questions (no timestamp, no question)",
               first_a == first_b),
        _check("First message holds the system prompt and the whole handbook",
               SYSTEM_PROMPT in first_a and handbook() in first_a),
        _check("First message is marked for caching (cache_control)", _has_cache_control(a[0])),
        _check("The question only appears in the last message",
               "What's the repo URL?" in _first_text(a[-1])
               and all("What's the repo URL?" not in _first_text(m) for m in a[:-1])),
        _check("Retrieved chunks go in the last message, after the cached prefix",
               "make dev" in _first_text(a[-1]) and "make dev" not in first_a),
    ]
    return all(results)


def live_check() -> None:
    print("\nC4 · live (gateway)\n")
    if not (config.LITELLM_PROXY_API_BASE and config.LITELLM_PROXY_API_KEY):
        print("⏭️  Skipped — set LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY to run it")
        return
    from src.cost_helper import call

    model = config.MID_MODEL
    first = call(build_messages("What's the repo URL?"), model, max_tokens=100)
    second = call(build_messages("Which port does the API run on?"), model, max_tokens=100)
    u1, u2 = first["usage"], second["usage"]
    print(f"  call 1: write {u1['cache_write_tokens']:>5} · read {u1['cache_read_tokens']:>5} · fresh {u1['input_tokens']}")
    print(f"  call 2: write {u2['cache_write_tokens']:>5} · read {u2['cache_read_tokens']:>5} · fresh {u2['input_tokens']}")
    if not _check("Second call read the prefix from the cache", u2["cache_read_tokens"] > 0):
        print("   Hints: is C4 done? Is the prefix above the model's minimum cacheable length?")


def main() -> int:
    ok = structure_checks()
    live_check()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
