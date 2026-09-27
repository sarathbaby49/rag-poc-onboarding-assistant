"""Layer 3 — Embedding primitives  (YOUR EXERCISE: E1).

The atoms underneath retrieval: turn text into a vector (`embed`, done for you),
and measure how close two vectors are (`cosine_similarity`, your exercise).

Implementing cosine yourself makes the whole "score = 1 - distance" idea concrete:
the number retrieval ranks by is just this function applied to the query and each
chunk.

Self-check:  python -m checks.check_embeddings
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def _encoder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


def embed(text: str) -> list[float]:
    """Return the (normalized) embedding vector for a piece of text.

    Done for you — this is exactly what ingest and retrieve use under the hood.
    """
    return _encoder().encode([text], normalize_embeddings=True)[0].tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """EXERCISE E1 — cosine similarity between two vectors, in the range -1..1.

        cosine = (a · b) / (|a| * |b|)

      - a · b  (the dot product) = sum(ai * bi over all i)
      - |a|    (the length)      = sqrt(sum(ai * ai))

    For normalized vectors |a| = |b| = 1, so this reduces to the dot product —
    but implement the full formula so it works for ANY two vectors.

    Self-check:  python -m checks.check_embeddings
    """
    # TODO(E1): implement the cosine formula above (pure Python is fine — no numpy needed).
    raise NotImplementedError("Exercise E1: implement cosine_similarity — see EXERCISES.md")
