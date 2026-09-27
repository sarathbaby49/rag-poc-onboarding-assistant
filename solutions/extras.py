"""REFERENCE SOLUTIONS for the extra exercises: I1, E1, R3.

Try each first! If you're stuck or out of time, copy the relevant function body
into the file named below.

  - I1  -> src/ingest.py   :: chunk_text_smart
  - E1  -> src/embeddings.py :: cosine_similarity
  - R3  -> src/retrieve.py :: confident_hits
"""

from __future__ import annotations

import math


# --- I1: boundary-aware chunking (src/ingest.py) -----------------------------
def chunk_text_smart(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        if end < len(text):
            # back up to the last whitespace so we don't cut a word/line
            cut = max(window.rfind(" "), window.rfind("\n"))
            if cut > 0:
                end = start + cut
                window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# --- E1: cosine similarity (src/embeddings.py) -------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --- R3: confidence threshold (src/retrieve.py) ------------------------------
# Inside src/retrieve.py this uses the module's own semantic_search:
#
#   def confident_hits(query, k=config.TOP_K, min_score=0.25):
#       return [h for h in semantic_search(query, k) if h["score"] >= min_score]
def confident_hits(semantic_search, query, k=4, min_score: float = 0.25) -> list[dict]:
    """Standalone form for reference; in retrieve.py drop the first arg and call
    the module-level semantic_search directly."""
    return [h for h in semantic_search(query, k) if h["score"] >= min_score]
