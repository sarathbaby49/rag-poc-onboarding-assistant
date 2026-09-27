# 🧭 Session Exercises — Retrieval & Memory

Hands-on tasks for the upskill session. You'll learn **retrieval** and **memory**
by *building* the two pieces that a RAG assistant needs before it can answer well.

> **No API key needed for anything on this page.** Retrieval runs a local
> embedding model; memory is plain Python. You only need a key later, for the
> separate *RAG generation* session.

---

## Setup (once, before the session)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.ingest          # builds the vector index → "Indexed 12 chunks"
```

Sanity check the index and get a feel for retrieval (no code yet):

```bash
python explore.py stats                          # what's in the store
python explore.py query "how do I log in?"       # semantic search, with scores
python explore.py compare "log in" "authenticate a user"
python explore.py keyword "getUserToken"         # keyword vs semantic → why hybrid
```

You edit two files: **`src/retrieve.py`** and **`src/memory.py`**.
You check your work with the two commands below. Stuck or out of time? The
answers are in **`solutions/`** — try first, then peek.

```bash
python -m checks.check_retrieval
python -m checks.check_memory
```

---

## ⏱️ Run of show (~50 min)

| Time | What |
|------|------|
| 0:00–0:10 | Concept: embeddings, vector store, top-k. Live-drive `explore.py`. |
| 0:10–0:20 | **R1** — implement `semantic_search` · self-check |
| 0:20–0:30 | Concept: session vs long-term memory, token budget. |
| 0:30–0:40 | **M1** — session window · **M2** — profile persistence · self-check |
| 0:40–0:48 | **R2 (stretch)** — hybrid search for fast finishers |
| 0:48–0:50 | Recap: how retrieval + memory feed the RAG-generation layer next. |

---

# Part A — Retrieval

### R1 (core) — Semantic search · ⏱️ 10 min
**File:** `src/retrieve.py` → `semantic_search()`
**Goal:** turn a question into a vector, find the nearest chunks, return them scored.

**Definition of done** (`python -m checks.check_retrieval`):
- returns a list of `k` `{text, source, score}` dicts, best first
- `score` is cosine similarity in `0..1` (`score = 1 - distance`)
- asking *"How do I set up my local environment?"* retrieves `setup.md`

**Hints:**
- `_encoder()` and `_collection()` are already written for you.
- `_encoder().encode([query], normalize_embeddings=True).tolist()` → query vector
- `_collection().query(query_embeddings=..., n_results=k)` returns parallel lists
  in `res["documents"][0]`, `res["metadatas"][0]`, `res["distances"][0]`.

### R2 (stretch) — Hybrid search · ⏱️ ~8 min
**File:** `src/retrieve.py` → `hybrid_search()`
**Goal:** blend BM25 keyword matching with semantic similarity so exact symbols
(like `getUserToken`) *and* paraphrases both rank well.

**Definition of done:** `getUserToken` surfaces `code/auth.py` in the top-k
(plain semantic often misses it — that's the motivation).

**Hint:** build a `rank_bm25.BM25Okapi` over all chunk texts, then combine the two
ranked lists with **reciprocal rank fusion**: `score += 1 / (60 + rank)` for each
list, then sort by the fused score. No weight tuning needed.

---

# Part B — Memory

### M1 (core) — Session window · ⏱️ 5 min
**File:** `src/memory.py` → `SessionMemory.as_messages()`
**Goal:** return only the last `self.window` turns so follow-up questions have
context without blowing the token budget.

**Definition of done** (`python -m checks.check_memory`):
- with `window=3` and 5 turns added, returns the last 3, in order.

**Hint:** one line — a list slice (`self.turns[-self.window:]`).

### M2 (core) — Profile persistence · ⏱️ 8 min
**File:** `src/memory.py` → `JoineeProfile.save()` and `.load()`
**Goal:** remember *who* the joinee is across restarts (role, completed steps) by
saving to a JSON file and loading it back.

**Definition of done:**
- `save()` writes a JSON file (e.g. `.profiles/<name>.json`)
- `JoineeProfile.load(name)` returns a profile with the same
  `name` / `role` / `completed_steps`
- (bonus) call `self.save()` inside `record_step()` so progress auto-persists.

**Hint:** `from src import config` gives `config.BASE_DIR`;
`json.dumps(...)` + `Path.write_text(...)` to save, `json.loads(Path.read_text())`
to load. Create the folder with `Path.mkdir(exist_ok=True)`.

---

## Where this goes next

Retrieval returns the right chunks; memory carries the conversation and the
person. The **RAG-generation** session (`src/rag.py`) plugs both into a grounded,
cited Claude answer. Other layers (agent, routing, eval) live in their own stubs —
see the table in `README.md`.

---

<!-- Other presenters: add your layer's exercises below this line. -->
