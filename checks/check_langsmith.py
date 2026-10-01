"""Self-check for the LangSmith exercises (LS1 config, LS2 tracing).

Run from the repo root:  python -m checks.check_langsmith
"""

from __future__ import annotations

import os
import sys


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- LS1a: LangSmith configuration ----------------------------------------
    print("-- LS1a: LangSmith configuration --")
    try:
        from src.langsmith_utils import ensure_langsmith_configured

        status = ensure_langsmith_configured()
        results.append(_check(
            "ensure_langsmith_configured returns a dict",
            isinstance(status, dict),
        ))
        results.append(_check(
            "status dict has required keys",
            all(k in status for k in ["tracing_enabled", "api_key_set", "project", "endpoint", "status"]),
        ))
        results.append(_check(
            "status is one of 'ok', 'misconfigured', 'disabled'",
            status.get("status") in ("ok", "misconfigured", "disabled"),
        ))

        # Check that the function correctly reads env vars
        tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        results.append(_check(
            f"tracing_enabled matches env (LANGCHAIN_TRACING_V2={'true' if tracing_v2 else 'false'})",
            status.get("tracing_enabled") == tracing_v2,
        ))

        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        api_key_set = bool(api_key) and api_key not in ("", "lsv2-...")
        results.append(_check(
            f"api_key_set matches env (key {'is' if api_key_set else 'is NOT'} set)",
            status.get("api_key_set") == api_key_set,
        ))

        # Warn if tracing is not actually enabled
        if not tracing_v2:
            print("\n   ⚠️  LANGCHAIN_TRACING_V2 is not 'true' in your .env")
            print("      Traces will NOT be sent. Set it to enable observability.")
        if not api_key_set:
            print("\n   ⚠️  LANGCHAIN_API_KEY is not set in your .env")
            print("      Sign up at https://smith.langchain.com/ and add your key.")

    except NotImplementedError:
        results.append(_check("ensure_langsmith_configured implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (LS1a)")
    except Exception as e:
        results.append(_check(f"ensure_langsmith_configured runs without error ({type(e).__name__}: {e})", False))

    # --- LS1b: LangSmith client -----------------------------------------------
    print("\n-- LS1b: LangSmith client --")
    try:
        from src.langsmith_utils import get_langsmith_client

        client = get_langsmith_client()
        results.append(_check(
            "get_langsmith_client returns an object",
            client is not None,
        ))

        # Check it's a LangSmith Client
        type_name = type(client).__name__
        results.append(_check(
            f"client is a LangSmith Client (got {type_name})",
            "Client" in type_name,
        ))
    except NotImplementedError:
        results.append(_check("get_langsmith_client implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (LS1b)")
    except ImportError as e:
        results.append(_check(f"langsmith is installed ({e})", False))
        print("   ↳ run: pip install langsmith")
    except Exception as e:
        results.append(_check(f"get_langsmith_client runs without error ({type(e).__name__}: {e})", False))

    # --- LS2a: Traced retrieval -----------------------------------------------
    print("\n-- LS2a: traced_retrieval (needs vector index) --")
    try:
        from src.langsmith_utils import traced_retrieval

        # Check it has the traceable wrapper
        is_traceable = (
            hasattr(traced_retrieval, "__wrapped__")
            or hasattr(traced_retrieval, "is_traceable")
            or "traceable" in str(getattr(traced_retrieval, "__qualname__", ""))
            or hasattr(traced_retrieval, "__langsmith_traceable__")
        )

        # Even if we can't detect the decorator, check it works
        try:
            hits = traced_retrieval("setup", k=2)
            results.append(_check(
                "traced_retrieval returns results",
                isinstance(hits, list) and len(hits) > 0,
            ))
            results.append(_check(
                "results have expected shape (text, source, score)",
                all({"text", "source", "score"} <= set(h) for h in hits),
            ))
        except Exception as e:
            results.append(_check(f"traced_retrieval runs without error ({type(e).__name__}: {e})", False))
            print("   ↳ make sure the vector index is built: python -m src.ingest")

    except NotImplementedError:
        results.append(_check("traced_retrieval implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (LS2a)")

    # --- LS2b: Traced format_context ------------------------------------------
    print("\n-- LS2b: traced_format_context --")
    try:
        from src.langsmith_utils import traced_format_context

        sample_hits = [
            {"text": "Setup instructions here.", "source": "setup.md", "score": 0.9},
            {"text": "Auth docs here.", "source": "auth.md", "score": 0.8},
        ]

        try:
            formatted = traced_format_context(sample_hits)
            results.append(_check(
                "traced_format_context returns a string",
                isinstance(formatted, str),
            ))
            results.append(_check(
                "formatted output contains source references",
                "setup.md" in formatted and "auth.md" in formatted,
            ))
            results.append(_check(
                "formatted output has numbered citations",
                "[1]" in formatted and "[2]" in formatted,
            ))
        except Exception as e:
            results.append(_check(f"traced_format_context runs without error ({type(e).__name__}: {e})", False))

    except NotImplementedError:
        results.append(_check("traced_format_context implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (LS2b)")

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
