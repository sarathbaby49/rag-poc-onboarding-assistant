"""REFERENCE SOLUTION for src/retrieve.py (Exercises R1 + R2).

Try the exercise first! If you're stuck or out of time, copy the relevant
function body into src/retrieve.py. Don't peek before you've had a go.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def _encoder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


# --- R1: semantic search -----------------------------------------------------
def semantic_search(query: str, k: int = config.TOP_K) -> list[dict]:
    query_vec = _encoder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=query_vec, n_results=k)
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": text, "source": meta["source"], "score": 1 - dist})
    return hits


# --- R2 (stretch): hybrid search via reciprocal rank fusion ------------------
@lru_cache(maxsize=1)
def _bm25_index():
    data = _collection().get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    tokenized = [d.lower().split() for d in docs]
    return BM25Okapi(tokenized), docs, metas


def hybrid_search(query: str, k: int = config.TOP_K) -> list[dict]:
    pool = max(k, 10)

    # 1. semantic ranking
    sem = semantic_search(query, k=pool)

    # 2. keyword (BM25) ranking
    bm25, docs, metas = _bm25_index()
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:pool]
    bm = [{"text": docs[i], "source": metas[i]["source"], "score": float(scores[i])} for i in order]

    # 3. reciprocal rank fusion (no weight tuning needed)
    C = 60
    fused: dict[str, dict] = {}
    for ranked in (sem, bm):
        for rank, hit in enumerate(ranked):
            key = hit["source"] + hit["text"][:40]
            slot = fused.setdefault(key, {"hit": hit, "score": 0.0})
            slot["score"] += 1.0 / (C + rank + 1)

    best = sorted(fused.values(), key=lambda x: -x["score"])
    return [x["hit"] for x in best[:k]]
