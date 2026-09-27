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
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def chunk_text_smart(text: str, size: int, overlap: int) -> list[str]:
    """EXERCISE I1 — chunk on a natural boundary instead of mid-word.

    `chunk_text` above slices at exactly `size` characters, so it can cut a word
    (or a line) in half — which blurs a chunk's meaning and its embedding. Improve
    it: still aim for ~`size` chars, but END each chunk at the last whitespace
    before the limit so words stay whole, and keep ~`overlap` chars of context
    between neighbours.

    Return a list of chunk strings (return [text] when text fits in one `size`).

    Hint: for each window `text[start:start+size]`, if it doesn't reach the end,
    find the last space with `window.rfind(" ")` (or `"\\n"`) and cut there; then
    advance `start` to `end - overlap`.

    To actually use it, point `build_index` at `chunk_text_smart` and re-ingest.

    Self-check:  python -m checks.check_ingest
    """
    # TODO(I1): implement boundary-aware chunking.
    raise NotImplementedError("Exercise I1: implement chunk_text_smart — see EXERCISES.md")


# --- 3. EMBED + 4. STORE -----------------------------------------------------

def build_index(chunker=chunk_text) -> None:
    print(f"Loading documents from {config.DATA_DIR} ...")
    docs = load_documents(config.DATA_DIR)

    # Split every document into chunks, carrying its source forward.
    # `chunker` defaults to the fixed-size splitter; the Retrieval Lab UI can pass
    # chunk_text_smart (exercise I1) to see boundary-aware chunking in action.
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
    encoder = SentenceTransformer(config.EMBED_MODEL)
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
