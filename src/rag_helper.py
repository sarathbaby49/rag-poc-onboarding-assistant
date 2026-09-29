"""Graceful wrappers around src.rag for the Streamlit UI (app.py).

Many RAG functions chain onto earlier exercises, so an unfinished one crashes
the whole page:

    answer                -> semantic_search (R1)
    answer_or_abstain     -> confident_hits (R3) -> semantic_search (R1)
    answer_conversational -> semantic_search (R1) + SessionMemory.as_messages (M1)

These wrappers catch NotImplementedError from any link in the chain and return a
normal {"answer", "sources"} dict whose "answer" says exactly which exercise to
finish next. The UI stays up and doubles as a live progress board — as you
implement R1, R3, M1, G1–G4, the friendly TODO messages turn into real answers.
"""

from __future__ import annotations

from src import rag


def _todo_result(exc: NotImplementedError) -> dict:
    """Shape an unfinished-exercise error like a normal answer so the UI renders."""
    return {
        "answer": (
            "⚠️ **This exercise isn't finished yet.**\n\n"
            f"> {exc}\n\n"
            "Implement it (or copy the reference from `solutions/`), then ask again."
        ),
        "sources": [],
    }


def safe_answer(question: str, **kwargs) -> dict:
    """answer() (G1), but a missing G1/R1 shows a TODO instead of crashing."""
    try:
        return rag.answer(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_or_abstain(question: str, **kwargs) -> dict:
    """answer_or_abstain() (G2), guarded against missing G2/R3/R1."""
    try:
        return rag.answer_or_abstain(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_conversational(question: str, memory, **kwargs) -> dict:
    """answer_conversational() (G4), guarded against missing G4/R1/M1."""
    try:
        return rag.answer_conversational(question, memory, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)
