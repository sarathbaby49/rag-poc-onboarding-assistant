"""Dependency shims that make the generation exercises (G1–G4) self-contained.

Students only write the minimal G code in src/rag.py. That code leans on earlier
sessions — semantic_search (R1), confident_hits (R3), SessionMemory (M1) — which
may not be done yet. So src/rag.py imports those names from HERE instead of from
src.retrieve / src.memory, and this module provides complete, working versions
(equivalent to the R1/R3/M1 solutions), built on the `_encoder()` / `_collection()`
helpers that ship done in src/retrieve.py.

Result: a student can write G1

    hits = semantic_search(question, k)
    text = _generate(question, hits)
    return {"answer": text, "sources": hits}

and it runs even though R1/R3/M1 are still stubs — the generation session stands
on its own.

This module also holds the crash-safe safe_* wrappers used by rag_app.py. They
import `rag` lazily (inside the functions) because src/rag.py imports this
module — a top-level `import rag` here would be a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import config
# _encoder()/_collection() are the "done for you" helpers from the R exercises.
from src.retrieve import _collection, _encoder


# --- R1: semantic (vector) search -------------------------------------------
def semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Working R1 so G1/G2/G4 run without the retrieval exercise being done."""
    query_vec = _encoder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=query_vec, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "score": 1 - dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


# --- R3: confidence filter --------------------------------------------------
def confident_hits(
    query: str, k: int = config.TOP_K, min_score: float = 0.25
) -> list[dict]:
    """Working R3 so G2's abstain path runs without the retrieval exercise."""
    return [h for h in semantic_search(query, k) if h["score"] >= min_score]


# --- M1: session memory window ----------------------------------------------
@dataclass
class SessionMemory:
    """Working M1 so G4's follow-ups run without the memory exercise being done."""

    window: int = 6
    turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def as_messages(self) -> list[dict]:
        return self.turns[-self.window:]


# --- crash-safe wrappers around the G functions (used by rag_app.py) --------
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
    """answer() (G1) — only a missing G1 shows a TODO; retrieval is provided here."""
    from src import rag  # lazy: rag imports this module (avoid circular import)

    try:
        return rag.answer(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_or_abstain(question: str, **kwargs) -> dict:
    """answer_or_abstain() (G2) — only a missing G2 shows a TODO."""
    from src import rag

    try:
        return rag.answer_or_abstain(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_conversational(question: str, memory, **kwargs) -> dict:
    """answer_conversational() (G4) — only a missing G4 shows a TODO."""
    from src import rag

    try:
        return rag.answer_conversational(question, memory, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_with_citations(question: str, k: int = config.TOP_K) -> dict:
    """G3 in action — generate (G1), then trim sources to the ones actually cited.

    used_sources(answer_text, hits) must filter the FULL retrieved set, so we
    re-fetch the raw hits here rather than reusing result["sources"], which a
    G3-wired answer() may have already trimmed (re-filtering a trimmed list would
    shift the [n] indices).

    Returns the usual {answer, sources} plus:
      - retrieved:  how many chunks were retrieved before filtering
      - g3_pending: None, or the TODO message if used_sources isn't implemented
    """
    from src import rag

    try:
        result = rag.answer(question, k=k)
    except NotImplementedError as exc:
        return {**_todo_result(exc), "retrieved": 0, "g3_pending": None}

    answer_text = result["answer"]
    raw_hits = semantic_search(question, k)  # full retrieved set
    try:
        cited = rag.used_sources(answer_text, raw_hits)
        g3_pending = None
    except NotImplementedError as exc:
        cited = raw_hits
        g3_pending = str(exc)

    return {
        "answer": answer_text,
        "sources": cited,
        "retrieved": len(raw_hits),
        "g3_pending": g3_pending,
    }
