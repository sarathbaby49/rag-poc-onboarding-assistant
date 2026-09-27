"""REFERENCE SOLUTION for src/rag.py (RAG generation via the LiteLLM gateway).

Try it yourself first. This shows the working generation step: retrieve chunks,
build a grounded prompt, ask the model through the gateway, return a cited answer.
No direct provider key needed — src.llm talks to the LiteLLM proxy.
"""

from __future__ import annotations

import sys

from src import config
from src.llm import complete
from src.retrieve import semantic_search

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
    """Return {"answer", "sources"} for a question, grounded in retrieved docs."""
    hits = semantic_search(question, k)
    context = format_context(hits)
    text = complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context sources:\n\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=config.MAX_TOKENS,
    )
    return {"answer": text, "sources": hits}


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
