"""Layer 3 — RAG answer generation  (the "G" in RAG).

    retrieve relevant chunks  ->  build a grounded prompt  ->  ask the model  ->  cite sources

============================================================================
STUB — owned by the RAG-FLOW presenter (a different session topic).
This file is intentionally left unimplemented in the baseline skeleton. It
depends on retrieval (src/retrieve.py, the Retrieval session), so it comes
together AFTER retrieval works. The scaffolding below (SYSTEM_PROMPT and
format_context) is left in place for whoever builds this layer.
============================================================================

The key idea when you build it: don't ask the model what it knows — hand it the
retrieved context and instruct it to answer ONLY from that, citing where each
fact came from. That's what makes answers trustworthy.
"""

from __future__ import annotations

import sys

from src import config
from src.retrieve import semantic_search

# The system prompt is the assistant's "constitution". Note the two rules that
# make RAG safe: answer only from context, and say so when the context is thin.
SYSTEM_PROMPT = """You are the Engineering Onboarding Assistant for a software team.
You help new joinees get productive: setup, architecture, codebase and process questions.

Rules:
- Answer ONLY using the numbered context sources provided by the user.
- Cite the sources you used inline like [1], [2] after the relevant sentence.
- If the context does not contain the answer, say so plainly and suggest who to ask.
- Be concise and practical. Prefer steps and commands over prose."""


def format_context(hits: list[dict]) -> str:
    """Turn retrieved chunks into a numbered block the model can cite by number."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Return {"answer", "sources"} for a question, grounded in retrieved docs.

    RAG-FLOW EXERCISE (owned by the generation presenter):
      1. hits = semantic_search(question, k)          # needs Retrieval (R1) done
      2. context = format_context(hits)
      3. Call the model via src.llm.complete (the LiteLLM gateway) with the
         SYSTEM_PROMPT + the context, instructing it to answer only from context
         and cite [n]. No direct provider key needed — the gateway handles auth.
      4. Return {"answer": <text>, "sources": hits}.

    Reference implementation: solutions/rag.py. Verify the gateway first with
    `python -m checks.check_llm`.
    """
    raise NotImplementedError(
        "RAG generation layer — owned by the RAG-flow presenter. "
        "Depends on retrieval (Exercise R1). See src/rag.py docstring."
    )


def _cli() -> None:
    question = " ".join(sys.argv[1:]) or "How do I set up my local environment?"
    result = answer(question)
    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, hit in enumerate(result["sources"], start=1):
        print(f"[{i}] {hit['source']}  (score {hit['score']:.2f})")


if __name__ == "__main__":
    _cli()
