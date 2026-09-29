"""Layer 3 — Embedding primitives  (YOUR EXERCISE: E1).

The atoms underneath retrieval: turn text into a vector (`embed`, done for you),
and measure how close two vectors are (`cosine_similarity`, your exercise).

Implementing cosine yourself makes the whole "score = 1 - distance" idea concrete:
the number retrieval ranks by is just this function applied to the query and each
chunk.

Self-check:  python -m checks.check_embeddings
"""

from __future__ import annotations

import math
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
    """Cosine similarity between two vectors, in the range -1..1.

        cosine = (a · b) / (|a| * |b|)

    For normalized vectors |a| = |b| = 1, so this reduces to the dot product, but
    the full formula works for any two vectors.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
