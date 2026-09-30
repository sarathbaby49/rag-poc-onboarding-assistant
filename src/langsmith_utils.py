"""Layer 4 — LangSmith tracing and observability (YOUR EXERCISES: LS1 + LS2).

LangSmith is how you **see inside** your agent and graph runs. Without it you're
flying blind — you don't know which tools were called, what the model received,
how long each step took, or why the answer was wrong.

This is not optional. Every agent run and graph execution should be traced.

Concepts covered:
  - Configuring LangSmith tracing (env vars + programmatic setup)
  - @traceable decorator for custom function tracing
  - Run metadata and tags for filtering traces
  - Viewing traces in the LangSmith dashboard

Self-check:  python -m checks.check_langsmith
"""

from __future__ import annotations

import os

from langsmith import Client, traceable

from src import config
from src.retrieve import semantic_search


# --- LS1: Configure and verify LangSmith tracing -----------------------------
# LangSmith tracing is configured via environment variables. LangChain and
# LangGraph auto-detect these and send traces. But you need to:
#   1. Ensure the env vars are set correctly
#   2. Verify the connection works
#   3. Understand what gets traced automatically vs what needs @traceable
#
# Self-check:  python -m checks.check_langsmith   (section LS1)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_langsmith_configured() -> dict:
    """EXERCISE LS1a — Verify LangSmith is configured and return the status.

    Check that the required environment variables are set:
      - LANGCHAIN_TRACING_V2 must be "true"
      - LANGCHAIN_API_KEY must be set (starts with "lsv2-" or "ls__")
      - LANGCHAIN_PROJECT should be set (default: "onboarding-assistant")

    Return a dict with:
      - "tracing_enabled": bool
      - "api_key_set": bool
      - "project": str (the project name)
      - "endpoint": str (the LangSmith API endpoint)
      - "status": "ok" | "misconfigured" | "disabled"

    Hint: use os.getenv() or config.LANGSMITH_* values.
    """
    # TODO(LS1a): check env vars and return status dict
    raise NotImplementedError("Exercise LS1a: verify LangSmith config — see EXERCISES.md")


def get_langsmith_client():
    """EXERCISE LS1b — Create and return a LangSmith Client instance.

    The Client lets you programmatically interact with LangSmith:
      - List runs/traces
      - Read feedback
      - Create datasets for evaluation

    Steps:
      1. Return Client() — it auto-reads LANGCHAIN_API_KEY from env

    Hint: wrap in try/except ImportError to give a helpful message if
    langsmith isn't installed.
    """
    # TODO(LS1b): create and return a LangSmith Client
    raise NotImplementedError("Exercise LS1b: create LangSmith client — see EXERCISES.md")


# --- LS2: Add custom tracing with @traceable ---------------------------------
# LangChain agents and LangGraph auto-trace. But your own functions (like
# retrieval, formatting, business logic) don't show up unless you mark them.
# The @traceable decorator adds any function to the trace tree.
#
# Self-check:  python -m checks.check_langsmith   (section LS2)
# ─────────────────────────────────────────────────────────────────────────────

def traced_retrieval(query: str, k: int = 4) -> list[dict]:
    """EXERCISE LS2a — Wrap semantic_search with @traceable for visibility.

    Decorate this function with @traceable so it appears in the LangSmith
    trace tree when called from an agent or graph node.

    Steps:
      1. Decorate with @traceable(run_type="retriever", name="semantic_search")
      2. Inside, call semantic_search(query, k) and return the results

    The run_type="retriever" tells LangSmith to render this as a retrieval
    step (with special UI for showing documents + scores).

    Hint:
      @traceable(run_type="retriever", name="semantic_search")
      def traced_retrieval(query, k=4):
          return semantic_search(query, k)
    """
    # TODO(LS2a): implement with @traceable decorator
    raise NotImplementedError("Exercise LS2a: add @traceable to retrieval — see EXERCISES.md")


def traced_format_context(hits: list[dict]) -> str:
    """EXERCISE LS2b — Wrap context formatting with @traceable.

    Decorate with @traceable(run_type="chain", name="format_context") so you
    can see exactly what context was assembled from the retrieved chunks.

    Steps:
      1. Decorate with @traceable
      2. Format hits into numbered context (same as rag.py's format_context)
      3. Return the formatted string

    Hint:
      @traceable(run_type="chain", name="format_context")
      def traced_format_context(hits):
          lines = []
          for i, hit in enumerate(hits, start=1):
              lines.append(f"[{i}] (source: {hit['source']})\\n{hit['text']}")
          return "\\n\\n".join(lines)
    """
    # TODO(LS2b): implement with @traceable decorator
    raise NotImplementedError("Exercise LS2b: add @traceable to formatting — see EXERCISES.md")


# --- CLI entry point ----------------------------------------------------------
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
    elif status["status"] == "disabled":
        print("\n❌ Tracing is disabled. Set LANGCHAIN_TRACING_V2=true in .env")
    else:
        print("\n❌ LangSmith is misconfigured. Check your .env — see EXERCISES.md (LS1)")


if __name__ == "__main__":
    _cli()
