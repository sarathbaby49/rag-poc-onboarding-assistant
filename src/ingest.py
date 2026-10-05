"""Layer 2/3 — Ingestion pipeline.

The whole job of RAG's "R" starts here:

    load documents  ->  split into chunks  ->  embed each chunk  ->  store vectors

Run once before the demo:

    python -m src.ingest

Re-run any time the docs change. It's idempotent (upsert by id).
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src import config

# --- 1. LOAD -----------------------------------------------------------------
# Turn every file in data/ into a list of (text, source_path) records.
# We keep it deliberately simple: read text, remember where it came from so we
# can cite it later. Real systems add PDF/Confluence/Slack loaders here.

TEXT_SUFFIXES = {".md", ".txt", ".py", ".js", ".ts", ".json", ".jsonl"}


def load_documents(data_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        rel = str(path.relative_to(config.BASE_DIR))
        if path.suffix == ".jsonl":
            # One record per line (e.g. tickets). Flatten each into readable text.
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    docs.append({"text": _record_to_text(obj), "source": f"{rel}#{i}"})
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            docs.append({"text": _record_to_text(obj), "source": rel})
        else:
            docs.append({"text": path.read_text(encoding="utf-8"), "source": rel})
    return docs


def _record_to_text(obj) -> str:
    """Flatten a JSON object/list into a plain-text blob the embedder can read."""
    if isinstance(obj, list):
        return "\n".join(_record_to_text(o) for o in obj)
    if isinstance(obj, dict):
        return "\n".join(f"{k}: {v}" for k, v in obj.items())
    return str(obj)


# --- 2. CHUNK ----------------------------------------------------------------
# Models retrieve *chunks*, not whole files. We slide a fixed window with
# overlap so an idea that straddles a boundary still lands whole in one chunk.

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Naive fixed-size splitter — slices at exactly `size` chars, so it can cut a
    word or line in half. Kept as a baseline to compare against chunk_text_smart
    (flip to it in the Retrieval Lab to see the difference)."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def chunk_text_smart(text: str, size: int, overlap: int) -> list[str]:
    """Boundary-aware splitter — the DEFAULT chunker used by build_index.

    Aims for ~`size` chars but ends each chunk at the last whitespace/newline
    before the limit, so words and lines stay whole (a cleaner chunk = a cleaner
    embedding). Keeps ~`overlap` chars of context between neighbours.
    """
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        if end < len(text):
            cut = max(window.rfind(" "), window.rfind("\n"))
            if cut > 0:
                end = start + cut
                window = text[start:end]
        piece = window.strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# --- 3. EMBED + 4. STORE -----------------------------------------------------

def build_index(chunker=chunk_text_smart) -> None:
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = load_documents(config.DATA_DIR)

    # Split every document into chunks, carrying its source forward.
    # `chunker` defaults to the boundary-aware splitter; the Retrieval Lab UI can
    # pass the naive fixed-size `chunk_text` to compare the two.
    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for doc in docs:
        for j, chunk in enumerate(chunker(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(chunk)
            metadatas.append({"source": doc["source"]})
            ids.append(f"{doc['source']}::chunk{j}")

    print(f"  {len(docs)} documents -> {len(chunks)} chunks")

    print(f"Embedding with {config.EMBED_MODEL} (first run downloads the model) ...")
    encoder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    # normalize_embeddings + cosine space => similarity scores land in an
    # intuitive 0..1 range (higher = more similar), nice for the demo.
    embeddings = encoder.encode(chunks, show_progress_bar=True, normalize_embeddings=True).tolist()

    print(f"Writing to vector store at {config.CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # Fresh collection each run keeps the demo reproducible.
    client.delete_collection(config.COLLECTION_NAME) if _exists(client) else None
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"Done. Indexed {collection.count()} chunks. Ready to answer questions.")


def _exists(client) -> bool:
    return any(c.name == config.COLLECTION_NAME for c in client.list_collections())


if __name__ == "__main__":
    build_index()
