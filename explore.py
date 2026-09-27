"""Explore the local vector database — a learning tool for retrieval.

No API key needed: everything here is embeddings + math, the "R" in RAG.

Run from the repo root after `python -m src.ingest`:

    python explore.py stats                       # what's in the store
    python explore.py show 5                       # one chunk + its vector
    python explore.py query "how do I log in?"     # retrieval, with scores
    python explore.py compare "log in" "authenticate a user"
    python explore.py neighbors 3                   # chunks most like chunk 3
    python explore.py keyword "getUserToken"       # keyword vs semantic (why hybrid)
    python explore.py map                           # 2D picture of the vectors

Each command prints a short "what you're seeing" note so you learn the concept,
not just the numbers.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def encoder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


def embed(text: str) -> np.ndarray:
    # normalize_embeddings=True matches ingest, so a dot product == cosine similarity.
    return encoder().encode([text], normalize_embeddings=True)[0]


def _all():
    """Every chunk with its vector, text and source (small dataset — fine to load all)."""
    d = collection().get(include=["embeddings", "documents", "metadatas"])
    return d["embeddings"], d["documents"], d["metadatas"]


def note(msg: str) -> None:
    print(f"\n\033[2m› {msg}\033[0m\n")


# --- stats -------------------------------------------------------------------
def cmd_stats():
    embs, docs, metas = _all()
    embs = np.array(embs)
    print(f"Collection: {config.COLLECTION_NAME}")
    print(f"Chunks stored: {len(docs)}")
    print(f"Embedding shape: {embs.shape}  (each chunk = {embs.shape[1]} numbers)")
    print(f"Vectors normalized? avg length = {np.linalg.norm(embs, axis=1).mean():.3f}  (1.0 = unit vectors)")
    print("\nidx  source                                        preview")
    print("-" * 78)
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        preview = doc.replace("\n", " ")[:38]
        print(f"{i:>3}  {meta['source']:<44}  {preview}")
    note("Each chunk is one point in 384-dimensional space. Retrieval = find the "
         "nearest points to your question. Use `show <idx>` to inspect one.")


# --- show one chunk ----------------------------------------------------------
def cmd_show(idx: int):
    embs, docs, metas = _all()
    vec = np.array(embs[idx])
    print(f"Chunk {idx}  —  source: {metas[idx]['source']}")
    print("-" * 78)
    print(docs[idx])
    print("-" * 78)
    print(f"Vector: {vec.shape[0]} dims, length {np.linalg.norm(vec):.3f}")
    print(f"First 8 of {vec.shape[0]} values: {np.round(vec[:8], 3).tolist()}")
    note("Those numbers ARE the meaning of this text to the model. On their own they "
         "mean nothing to us — they only matter relative to other chunks' vectors.")


# --- query (the core of retrieval) -------------------------------------------
def cmd_query(text: str, k: int = 4):
    qv = embed(text)
    res = collection().query(query_embeddings=[qv.tolist()], n_results=k,
                             include=["documents", "metadatas", "distances"])
    print(f'Query: "{text}"\n')
    print("rank  score  cos.dist  source")
    print("-" * 70)
    for r, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
    ):
        print(f"{r:>3}   {1 - dist:>5.2f}   {dist:>6.3f}   {meta['source']}")
    top = res["documents"][0][0].replace("\n", " ")[:120]
    print(f'\ntop chunk: "{top}..."')
    note("score = 1 - cosine_distance. Higher = closer in meaning. This is exactly "
         "what src/rag.py feeds to Claude as context. Change the wording and watch "
         "the ranking move — that's semantic search.")


# --- compare two strings -----------------------------------------------------
def cmd_compare(a: str, b: str):
    sim = float(embed(a) @ embed(b))   # dot product of unit vectors = cosine similarity
    print(f'"{a}"')
    print(f'"{b}"')
    print(f"\ncosine similarity = {sim:.3f}   (1 = identical meaning, 0 = unrelated)")
    note("Try synonyms ('log in' vs 'authenticate') vs unrelated pairs ('log in' vs "
         "'refund money'). Meaning, not shared words, drives the number.")


# --- neighbors of a chunk ----------------------------------------------------
def cmd_neighbors(idx: int, k: int = 4):
    embs, docs, metas = _all()
    embs = np.array(embs)
    sims = embs @ embs[idx]                 # cosine sim to every chunk (vectors are unit)
    order = np.argsort(-sims)
    print(f"Chunks most similar to #{idx} ({metas[idx]['source']}):\n")
    print("sim   idx  source")
    print("-" * 60)
    for j in order[:k + 1]:
        mark = "  <- itself" if j == idx else ""
        print(f"{sims[j]:.2f}  {j:>3}  {metas[j]['source']}{mark}")
    note("Chunks from the same doc/topic cluster together. This is why chunking and "
         "which docs you ingest matter so much — clusters are what retrieval walks.")


# --- keyword vs semantic (motivates hybrid search) ---------------------------
def cmd_keyword(text: str, k: int = 4):
    embs, docs, metas = _all()
    terms = text.lower().split()
    # naive keyword score: how many query terms appear in the chunk
    kw = [(sum(t in doc.lower() for t in terms), i) for i, doc in enumerate(docs)]
    kw.sort(reverse=True)
    # semantic score
    qv = embed(text)
    sims = np.array(embs) @ qv
    sem = sorted(enumerate(sims), key=lambda x: -x[1])

    def short(src):  # drop the common data dir so columns line up
        return src.replace("data/sample_company/", "")

    print(f'Query: "{text}"\n')
    print(f"{'KEYWORD (term hits)':<40}SEMANTIC (cosine)")
    print("-" * 62)
    for r in range(k):
        ki, si = kw[r][1], sem[r][0]
        left = f"{kw[r][0]} hit  {short(metas[ki]['source'])}"
        right = f"{sem[r][1]:.2f}  {short(metas[si]['source'])}"
        print(f"{left:<40}{right}")
    note("Keyword nails exact tokens (code symbols, error strings) but misses "
         "paraphrases. Semantic is the reverse. Blending them = hybrid search "
         "(the Layer 3 exercise in src/retrieve.py).")


# --- 2D map of the vectors ---------------------------------------------------
def cmd_map():
    embs, docs, metas = _all()
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(np.array(embs))
    # assign each source file a legend letter
    sources = sorted({m["source"].split("#")[0] for m in metas})
    letter = {s: chr(ord("A") + i) for i, s in enumerate(sources)}
    W, H = 62, 22
    xs, ys = xy[:, 0], xy[:, 1]
    gx = ((xs - xs.min()) / (np.ptp(xs) + 1e-9) * (W - 1)).astype(int)
    gy = ((ys - ys.min()) / (np.ptp(ys) + 1e-9) * (H - 1)).astype(int)
    grid = [[" "] * W for _ in range(H)]
    for i, m in enumerate(metas):
        grid[H - 1 - gy[i]][gx[i]] = letter[m["source"].split("#")[0]]
    print("Chunks projected from 384 dims down to 2 (PCA):\n")
    print("+" + "-" * W + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * W + "+")
    print("\nLegend:")
    for s, c in letter.items():
        print(f"  {c} = {s}")
    note("Points that sit together are similar in meaning. Same-topic chunks form "
         "clusters — retrieval just returns the cluster nearest your question.")


COMMANDS = {
    "stats": lambda a: cmd_stats(),
    "show": lambda a: cmd_show(int(a[0])),
    "query": lambda a: cmd_query(" ".join(a)),
    "compare": lambda a: cmd_compare(a[0], a[1]),
    "neighbors": lambda a: cmd_neighbors(int(a[0])),
    "keyword": lambda a: cmd_keyword(" ".join(a)),
    "map": lambda a: cmd_map(),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])
