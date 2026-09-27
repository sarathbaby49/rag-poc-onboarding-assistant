"""Layer 3 — Retrieval  (YOUR EXERCISES: R1 core, R2 stretch).

Given a question, find the most relevant chunks from the vector store.

  - R1 (core):    implement `semantic_search`  -> see EXERCISES.md
  - R2 (stretch): implement `hybrid_search`    -> see EXERCISES.md
  - R3 (extra):   implement `confident_hits`   -> see EXERCISES.md

The `_encoder()` and `_collection()` helpers below are done for you.
Self-check your work:  python -m checks.check_retrieval
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
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
    """EXERCISE R1 — semantic (vector) search.

    Return a list of {"text", "source", "score"} dicts, best first.

    Steps:
      1. Embed the query into a vector:
             query_vec = _encoder().encode([query], normalize_embeddings=True).tolist()
      2. Ask the vector store for the k nearest chunks:
             res = _collection().query(query_embeddings=query_vec, n_results=k)
      3. Build one dict per hit by zipping these three parallel lists:
             res["documents"][0], res["metadatas"][0], res["distances"][0]
         The store returns cosine *distance*; turn it into a similarity with
             score = 1 - dist
         and read the source from meta["source"].

    Self-check:  python -m checks.check_retrieval
    """
    # TODO(R1): delete the line below and implement the three steps above.
    raise NotImplementedError("Exercise R1: implement semantic_search — see EXERCISES.md")


def hybrid_search(query: str, k: int = config.TOP_K) -> list[dict]:
    """EXERCISE R2 (stretch) — blend BM25 keyword search with semantic search.

    Why: vector search is great at meaning ("how do I log in?") but weak at exact
    symbols ("getUserToken"). BM25 (keyword) is the opposite. Blending them —
    'hybrid search' — catches both.

    Your mission:
      1. Build a BM25 index over all chunks (rank_bm25.BM25Okapi over
         _collection().get(include=["documents", "metadatas"])).
      2. Score chunks by BM25 for the query's keywords.
      3. Combine with the semantic ranking from `semantic_search`. The simplest
         robust blend is reciprocal rank fusion: score = sum(1 / (C + rank)) over
         both ranked lists (C≈60), then sort by the fused score.
      4. Return the top-k.
    """
    # TODO(R2): replace this fallback with a real BM25 + semantic blend.
    raise NotImplementedError("Exercise R2: implement hybrid_search — see EXERCISES.md")


def confident_hits(query: str, k: int = config.TOP_K, min_score: float = 0.25) -> list[dict]:
    """EXERCISE R3 (extra) — retrieve, then drop weak matches below `min_score`.

    Semantic search always returns k results, even for an off-topic question — so
    the assistant would "answer" from irrelevant context. Filtering by score lets
    it honestly say "I don't know" when nothing is relevant enough.

    Steps:
      1. hits = semantic_search(query, k)          # needs R1 done
      2. keep only hits whose "score" >= min_score
      3. return the filtered list (it may be empty)

    Self-check:  python -m checks.check_retrieval
    """
    # TODO(R3): filter semantic_search results by min_score.
    raise NotImplementedError("Exercise R3: implement confident_hits — see EXERCISES.md")
