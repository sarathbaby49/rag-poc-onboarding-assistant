"""REFERENCE SOLUTION for src/rag.py — the "G" in RAG  (Exercises G1–G4).

Try each yourself first! If you're stuck or out of time, copy the relevant
function body into src/rag.py. No direct provider key needed — src.llm talks to
the LiteLLM proxy (verify it with `python -m checks.check_llm`).

  - G1  -> answer                 (retrieve → grounded, cited answer)
  - G2  -> answer_or_abstain      (honest "I don't know")
  - G3  -> used_sources           (return only the sources actually cited)
  - G4  -> answer_conversational  (thread session memory for follow-ups)
"""

from __future__ import annotations

import re
import sys

from src import config
from src.llm import complete
from src.memory import SessionMemory
from src.retrieve import confident_hits, semantic_search

SYSTEM_PROMPT = """You are the Engineering Onboarding Assistant for a software team.
You help new joinees get productive: setup, architecture, codebase and process questions.

Rules:
- Answer ONLY using the numbered context sources provided by the user.
- Cite the sources you used inline like [1], [2] after the relevant sentence.
- If the context does not contain the answer, say so plainly and suggest who to ask.
- Be concise and practical. Prefer steps and commands over prose."""

ABSTAIN_MESSAGE = (
    "I couldn't find this in the onboarding docs. "
    "Try asking in #eng-help on Slack, or check with your onboarding buddy."
)


def format_context(hits: list[dict]) -> str:
    """Turn retrieved chunks into a numbered block the model can cite by number."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)


def _generate(question: str, hits: list[dict]) -> str:
    """Shared generation step: hand the model the context + question, get text back."""
    context = format_context(hits)
    return complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context sources:\n\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=config.MAX_TOKENS,
    )


# --- G1: grounded generation -------------------------------------------------
def answer(question: str, k: int = config.TOP_K) -> dict:
    hits = semantic_search(question, k)
    text = _generate(question, hits)
    return {"answer": text, "sources": hits}


# --- G2: honest "I don't know" ----------------------------------------------
def answer_or_abstain(question: str, k: int = config.TOP_K, min_score: float = 0.25) -> dict:
    hits = confident_hits(question, k, min_score)   # R3: drops weak matches
    if not hits:
        # Nothing cleared the bar — hand off without spending a model call.
        return {"answer": ABSTAIN_MESSAGE, "sources": []}
    text = _generate(question, hits)
    return {"answer": text, "sources": used_sources(text, hits) or hits}


# --- G3: trustworthy citations ----------------------------------------------
def used_sources(answer_text: str, hits: list[dict]) -> list[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    return [h for i, h in enumerate(hits, start=1) if i in cited]


# --- G4: conversational RAG (memory) ----------------------------------------
def answer_conversational(question: str, memory: SessionMemory, k: int = config.TOP_K) -> dict:
    hits = semantic_search(question, k)
    context = format_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *memory.as_messages(),   # M1: the recent window keeps follow-ups cheap
        {"role": "user", "content": f"Context sources:\n\n{context}\n\nQuestion: {question}"},
    ]
    text = complete(messages, max_tokens=config.MAX_TOKENS)
    memory.add("user", question)
    memory.add("assistant", text)
    return {"answer": text, "sources": hits}


def _cli() -> None:
    question = " ".join(sys.argv[1:]) or "How do I set up my local environment?"
    result = answer_or_abstain(question)
    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, hit in enumerate(result["sources"], start=1):
        print(f"[{i}] {hit['source']}  (score {hit['score']:.2f})")


if __name__ == "__main__":
    _cli()
