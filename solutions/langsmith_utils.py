"""REFERENCE SOLUTION for src/langsmith_utils.py (Exercises LS1 + LS2).

LangSmith tracing and observability utilities.
Try the exercise first! If you're stuck or out of time, copy the relevant
function into src/langsmith_utils.py.
"""

from __future__ import annotations

import os

from langsmith import Client, traceable

from src import config
from src.retrieve import semantic_search


# --- LS1: Configure and verify ------------------------------------------------

def ensure_langsmith_configured() -> dict:
    """Verify LangSmith is configured and return the status."""
    tracing_enabled = config.LANGSMITH_TRACING_ENABLED
    api_key = config.LANGSMITH_API_KEY
    project = config.LANGSMITH_PROJECT
    endpoint = config.LANGSMITH_ENDPOINT

    api_key_set = bool(api_key) and api_key != "lsv2-..."

    if not tracing_enabled:
        status = "disabled"
    elif not api_key_set:
        status = "misconfigured"
    else:
        status = "ok"

    return {
        "tracing_enabled": tracing_enabled,
        "api_key_set": api_key_set,
        "project": project,
        "endpoint": endpoint,
        "status": status,
    }


def get_langsmith_client() -> Client:
    """Create and return a LangSmith Client instance."""
    try:
        return Client()
    except ImportError:
        raise ImportError(
            "langsmith is not installed. Run: pip install langsmith"
        )


# --- LS2: Custom tracing with @traceable --------------------------------------

@traceable(run_type="retriever", name="semantic_search")
def traced_retrieval(query: str, k: int = 4) -> list[dict]:
    """Semantic search wrapped with @traceable for LangSmith visibility."""
    return semantic_search(query, k)


@traceable(run_type="chain", name="format_context")
def traced_format_context(hits: list[dict]) -> str:
    """Format retrieved chunks into numbered context, traced in LangSmith."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)


# --- CLI ----------------------------------------------------------------------
def _cli() -> None:
    """Quick check: print LangSmith configuration status."""
    status = ensure_langsmith_configured()
    print("🔍 LangSmith Configuration:")
    print(f"   Tracing enabled:  {status['tracing_enabled']}")
    print(f"   API key set:      {status['api_key_set']}")
    print(f"   Project:          {status['project']}")
    print(f"   Endpoint:         {status['endpoint']}")
    print(f"   Status:           {status['status']}")

    if status["status"] == "ok":
        print("\n✅ LangSmith is properly configured. Traces will appear at:")
        print(f"   https://smith.langchain.com/o/default/projects/p/{status['project']}")

        # Verify connectivity
        try:
            client = get_langsmith_client()
            # A lightweight call to check connectivity
            print("✅ LangSmith client created successfully.")
        except Exception as e:
            print(f"⚠️  Client created but connectivity check failed: {e}")
    elif status["status"] == "disabled":
        print("\n❌ Tracing is disabled. Set LANGCHAIN_TRACING_V2=true in .env")
    else:
        print("\n❌ LangSmith is misconfigured. Check your .env — see EXERCISES.md (LS1)")


if __name__ == "__main__":
    _cli()
