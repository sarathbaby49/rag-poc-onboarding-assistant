"""Connectivity check for the LiteLLM gateway.

Confirms the proxy base URL + key in .env work, independent of any exercise.
Run from the repo root:  python -m checks.check_llm
"""

from __future__ import annotations

import sys

from src import config
from src.llm import complete


def main() -> None:
    print(f"Model: {config.LLM_MODEL}")
    print(f"Proxy: {config.LITELLM_PROXY_API_BASE or '(provider default)'}")

    if config.LLM_MODEL.startswith("litellm_proxy/") and not config.LITELLM_PROXY_API_KEY:
        print("❌ LITELLM_PROXY_API_KEY not set — copy .env.example to .env and fill it in.")
        sys.exit(1)

    try:
        reply = complete(
            [{"role": "user", "content": "Reply with exactly one word: pong"}],
            max_tokens=16,
        )
    except Exception as e:  # noqa: BLE001
        print(f"❌ Gateway call failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"✅ Gateway reachable. Model replied: {reply.strip()!r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
