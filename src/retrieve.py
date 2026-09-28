"""Layer 3 — Retrieval.

Given a question, find the most relevant chunks from the vector store.

  - semantic_search : embed the query, return the k nearest chunks (cosine)
  - hybrid_search   : blend BM25 keyword ranking with semantic (reciprocal rank fusion)
  - confident_hits  : semantic_search + a min-score threshold ("I don't know" support)

Self-check:  python -m checks.check_retrieval
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def _encoder() -> SentenceTransformer:
    # Cached so we load the embedding model into memory only once.
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Embed the query and return the k nearest chunks by cosine similarity.

    Returns a list of {"text", "source", "score"} dicts, best first, where
    score = 1 - cosine_distance (so it reads 0..1, higher = closer).
    """
    query_vec = _encoder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=query_vec, n_results=k)
    return [
        {"text": text, "source": meta["source"], "score": 1 - dist}
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


@lru_cache(maxsize=1)
def _bm25_index():
    """BM25 index over every stored chunk (built once)."""
    data = _collection().get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    tokenized = [d.lower().split() for d in docs]
    return BM25Okapi(tokenized), docs, metas


def hybrid_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Blend BM25 keyword search with semantic search via reciprocal rank fusion.

    Semantic search is strong on meaning but weak on exact symbols; BM25 is the
    reverse. RRF combines the two ranked lists without weight tuning.
    """
    pool = max(k, 10)

    # 1. semantic ranking
    sem = semantic_search(query, k=pool)

    # 2. keyword (BM25) ranking
    bm25, docs, metas = _bm25_index()
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:pool]
    bm = [{"text": docs[i], "source": metas[i]["source"], "score": float(scores[i])} for i in order]

    # 3. reciprocal rank fusion: score = sum(1 / (C + rank)) across both lists
    C = 60
    fused: dict[str, dict] = {}
    for ranked in (sem, bm):
        for rank, hit in enumerate(ranked):
            key = hit["source"] + hit["text"][:40]
            slot = fused.setdefault(key, {"hit": hit, "score": 0.0})
            slot["score"] += 1.0 / (C + rank + 1)

    best = sorted(fused.values(), key=lambda x: -x["score"])
    return [x["hit"] for x in best[:k]]


def confident_hits(query: str, k: int = config.TOP_K, min_score: float = 0.25) -> list[dict]:
    """Retrieve, then drop weak matches below `min_score`.

    Lets the assistant honestly return nothing (→ "I don't know") when no chunk
    is relevant enough, instead of answering from irrelevant context.
    """
    return [h for h in semantic_search(query, k) if h["score"] >= min_score]
