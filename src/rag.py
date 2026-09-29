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
from typing import Callable

from src import config
from src.llm import complete
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


def answer(
    question: str,
    k: int = config.TOP_K,
    *,
    history: list[dict] | None = None,
    memories: list[str] | None = None,
    recalled: list[str] | None = None,
    retriever: Callable[[str, int], list[dict]] = semantic_search,
    generate: Callable[..., str] = complete,
) -> dict:
    """Return {"answer", "sources", "messages"} for a question, grounded in docs.

    `messages` is the full OpenAI-style payload (the context window) that was sent
    to the model — system prompt, recalled memories, session history and the
    context-stuffed user turn — so callers can inspect exactly what the LLM saw.

    The RAG flow: retrieve relevant chunks -> build a grounded prompt -> ask the
    model through the LiteLLM gateway -> return a cited answer + its sources.

    Memory wiring (optional, so single-shot callers like app.py stay simple):
      - `history`  : prior conversation turns ({"role", "content"}) — session
                     memory — so follow-ups ("what about the tests?") have context.
      - `memories` : stable facts about the person (e.g. a joinee profile),
                     injected as a "what you already know" system message.
      - `recalled` : long-term memories pulled by MEANING for this question
                     (MemoryStore.recall), injected as their OWN labeled system
                     message so they're distinct from the profile in the context
                     window.
      - `retriever`: which retrieval function to use (semantic_search by default;
                     pass hybrid_search to blend in keyword matching).
      - `generate` : which generator to call (the LiteLLM gateway `complete` by
                     default).
    """
    hits = retriever(question, k)
    context = format_context(hits)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memories:
        known = "\n".join(f"- {m}" for m in memories)
        messages.append({
            "role": "system",
            "content": f"What you already know about this person:\n{known}",
        })
    if recalled:
        remembered = "\n".join(f"- {m}" for m in recalled)
        messages.append({
            "role": "system",
            "content": "Recalled from earlier (by relevance) — past things this "
                       f"person said that relate to their question:\n{remembered}",
        })
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context sources:\n\n{context}\n\nQuestion: {question}",
    })

    text = generate(messages, max_tokens=config.MAX_TOKENS)
    # `messages` is the exact context window we hand the model — return it so
    # callers (e.g. the Lab UI) can show what was actually sent.
    return {"answer": text, "sources": hits, "messages": messages}


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
