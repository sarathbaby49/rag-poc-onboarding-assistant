"""Make the generation exercises (G1–G4) runnable on their own, for rag_app.py.

The G exercises in src/rag.py lean on earlier sessions:

    answer                -> semantic_search (R1)
    answer_or_abstain     -> confident_hits (R3) -> semantic_search (R1)
    answer_conversational -> semantic_search (R1) + SessionMemory.as_messages (M1)

But the generation session is meant to be **independent** — you shouldn't need to
finish the retrieval/memory sessions first. So this module provides small,
self-contained versions of those dependencies (built on the `_encoder()` /
`_collection()` helpers that ship done in src/retrieve.py — no solutions/ import)
and uses them ONLY as a fallback: if you've implemented R1/R3/M1 yourself, your
code runs; if you haven't, the fallback fills in so your G1–G4 code still works.

We inject the fallbacks into the `rag` module so its internal calls pick them up,
and expose a SessionMemory whose window works out of the box for G4. The
safe_* wrappers then only have to catch a missing G function itself (G1–G4),
turning it into a friendly "implement this next" message instead of a crash.
"""

from __future__ import annotations

from src import config, rag, retrieve
from src.memory import SessionMemory as _SessionMemory
# _encoder()/_collection() are the "done for you" helpers from the R exercises.
from src.retrieve import _collection, _encoder


# --- self-contained dependency fallbacks ------------------------------------
def _fallback_semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Stand-in for R1 so G1/G2/G4 work before semantic_search is implemented."""
    query_vec = _encoder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=query_vec, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "score": 1 - dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Prefer the participant's R1; fall back to the built-in version."""
    try:
        return retrieve.semantic_search(query, k)
    except NotImplementedError:
        return _fallback_semantic_search(query, k)


def _confident_hits(
    query: str, k: int = config.TOP_K, min_score: float = 0.25
) -> list[dict]:
    """Prefer the participant's R3; fall back to filtering _semantic_search."""
    try:
        return retrieve.confident_hits(query, k, min_score)
    except NotImplementedError:
        return [h for h in _semantic_search(query, k) if h["score"] >= min_score]


# Inject the fallbacks into the rag module so answer()/answer_or_abstain() pick
# them up (rag.py did `from src.retrieve import semantic_search, confident_hits`,
# so these names live in the rag namespace).
rag.semantic_search = _semantic_search
rag.confident_hits = _confident_hits


class SessionMemory(_SessionMemory):
    """SessionMemory whose window works even before M1 is implemented."""

    def as_messages(self) -> list[dict]:
        try:
            return super().as_messages()
        except NotImplementedError:
            return self.turns[-self.window:]


# --- crash-safe wrappers around the G functions themselves ------------------
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
    """answer() (G1); dependencies are covered, so only a missing G1 shows a TODO."""
    try:
        return rag.answer(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_or_abstain(question: str, **kwargs) -> dict:
    """answer_or_abstain() (G2); only a missing G2 shows a TODO."""
    try:
        return rag.answer_or_abstain(question, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_conversational(question: str, memory, **kwargs) -> dict:
    """answer_conversational() (G4); only a missing G4 shows a TODO."""
    try:
        return rag.answer_conversational(question, memory, **kwargs)
    except NotImplementedError as exc:
        return _todo_result(exc)


def safe_answer_with_citations(question: str, k: int = config.TOP_K) -> dict:
    """G3 in action — generate (G1), then trim sources to the ones actually cited.

    used_sources(answer_text, hits) must filter the FULL retrieved set, so we
    re-fetch the raw hits here (order is deterministic for the same query + k)
    rather than reusing result["sources"], which a G3-wired answer() may have
    already trimmed — re-filtering a trimmed list would shift the [n] indices.

    Returns the usual {answer, sources} plus:
      - retrieved:  how many chunks were retrieved before filtering
      - g3_pending: None, or the TODO message if used_sources isn't implemented
    """
    try:
        result = rag.answer(question, k=k)
    except NotImplementedError as exc:
        return {**_todo_result(exc), "retrieved": 0, "g3_pending": None}

    answer_text = result["answer"]
    raw_hits = _semantic_search(question, k)  # full retrieved set (fallback-safe)
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
