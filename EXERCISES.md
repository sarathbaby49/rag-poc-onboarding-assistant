# 🧭 Session Exercises — Ingestion, Embeddings, Retrieval (+ Memory)

Hands-on tasks for the upskill session. You learn the "R" in RAG — how docs
become searchable meaning and how the right pieces come back — by *building* each
piece yourself.

> **No API key needed for anything on this page.** Everything here runs on a
> local embedding model + plain Python. (A key/gateway is only for the separate
> *RAG generation* session.)

Exercises are a **menu**, not a checklist. **R1 is the anchor** (everything leans
on it); the rest add depth — pick what fits your time. Difficulty: 🟢 easy ·
🟡 medium · 🔵 explore.

---

## Setup (once, before the session)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.ingest          # builds the vector index → "Indexed 57 chunks"
```

The sample company is **Acme, an e-commerce platform** — the corpus has service
code (catalog, cart, orders, inventory, checkout, payments), architecture &
onboarding docs, Slack channels, tickets, and a sales report (~57 chunks).

Get a feel for retrieval first (no code yet):

```bash
python explore.py stats                          # what's in the store
python explore.py query "how does checkout work?"      # semantic search, with scores
python explore.py query "what were our Q2 sales?"      # finds the sales report
python explore.py compare "log in" "authenticate a user"
python explore.py keyword "getUserToken"         # keyword vs semantic → why hybrid
```

**Files you edit:** `src/ingest.py`, `src/embeddings.py`, `src/retrieve.py`,
`src/memory.py`. **Self-check any exercise** with its command below (instant
✅/❌). Stuck or out of time? Answers are in **`solutions/`** — try first, then peek.

```bash
python -m checks.check_ingest        # I1
python -m checks.check_embeddings    # E1
python -m checks.check_retrieval     # R1 (+ R2, R3 reported)
python -m checks.check_memory        # M1, M2
```

---

## ⏱️ Run of show (~50 min)

| Time | What |
|------|------|
| 0:00–0:10 | Concept: embeddings, cosine, ingest → store → retrieve. Live-drive `explore.py`. |
| 0:10–0:22 | **R1** — `semantic_search` (everyone) · self-check |
| 0:22–0:38 | Pick your depth: **E1** cosine · **I1** smarter chunking · **R3** confidence · **E2** swap model |
| 0:38–0:48 | Memory: **M1** session window · **M2** profile persistence |
| 0:48–0:50 | Recap: how it all feeds the RAG-generation layer next. |

Fast finishers go for **R2** (hybrid search). Nobody needs to finish everything.

---

# Part A — Ingestion

### I1 (medium 🟡) — Boundary-aware chunking · ⏱️ ~12 min
**File:** `src/ingest.py` → `chunk_text_smart()`
**Goal:** split text at natural boundaries (spaces/newlines) instead of mid-word,
so each chunk stays coherent. Chunking is the single biggest retrieval-quality
lever — this is where you feel it.

**Definition of done** (`python -m checks.check_ingest`):
- returns multiple chunks, each `<= size`
- never cuts a word in half
- text shorter than `size` → a single chunk

**Hint:** for each window `text[start:start+size]`, if it isn't the end, back up
to the last `" "` (or `"\n"`) with `rfind`, cut there, then advance
`start = end - overlap`. To use it for real: point `build_index` at
`chunk_text_smart` and re-run `python -m src.ingest`, then `explore.py query` to
compare.

---

# Part B — Embeddings

### E1 (easy 🟢) — Cosine similarity by hand · ⏱️ ~8 min
**File:** `src/embeddings.py` → `cosine_similarity()`
**Goal:** compute the number retrieval actually ranks by, so `score = 1 − distance`
stops being magic. `embed()` is provided.

**Definition of done** (`python -m checks.check_embeddings`):
- identical vectors → `1.0`; perpendicular → `0.0`
- works on un-normalized vectors (`[1,1]` vs `[2,2]` → `1.0`)
- a related pair (`"log in"` / `"authenticate a user"`) scores higher than an
  unrelated pair (`"log in"` / `"refund a payment"`)

**Hint:** `cosine = dot(a,b) / (|a|·|b|)`; `dot = sum(x*y)`, `|a| = sqrt(sum(x*x))`.
Pure Python is fine.

### E2 (explore 🔵) — Swap the embedding model · ⏱️ ~8 min
**File:** `src/config.py` → `EMBED_MODEL`
**Goal:** see how model choice changes retrieval. Change `EMBED_MODEL` to another
sentence-transformers model (e.g. `all-MiniLM-L12-v2` or `paraphrase-MiniLM-L3-v2`),
then **re-ingest** and compare.

**What to observe (no self-check — this is exploratory):**
- `python -m src.ingest` then `python explore.py stats` — did the vector
  dimension change?
- `python explore.py query "how do I log in?"` — did the ranking or scores move?
- Why you *must* re-ingest: vectors from different models live in different
  spaces and aren't comparable, so the index has to be rebuilt.

---

# Part C — Retrieval

### R1 (core 🟢) — Semantic search · ⏱️ ~12 min
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

### R3 (extra 🟡) — Confidence threshold / "I don't know" · ⏱️ ~8 min
**File:** `src/retrieve.py` → `confident_hits()`  · *(needs R1 done)*
**Goal:** drop weak matches so an off-topic question returns *nothing* instead of
forcing the model to answer from irrelevant context. This is how a RAG assistant
honestly says "I don't know."

**Definition of done** (reported by `python -m checks.check_retrieval`):
- `min_score=0.0` keeps everything `semantic_search` returned
- a high `min_score` (e.g. `0.99`) returns an empty list

**Hint:** one comprehension — `[h for h in semantic_search(query, k) if h["score"] >= min_score]`.

### R2 (stretch 🔵) — Hybrid search · ⏱️ ~10 min
**File:** `src/retrieve.py` → `hybrid_search()`
**Goal:** blend BM25 keyword matching with semantic similarity so exact symbols
(like `getUserToken`) *and* paraphrases both rank well.

**Definition of done:** `getUserToken` surfaces `code/auth.py` in the top-k
(plain semantic often misses it — that's the motivation).

**Hint:** build a `rank_bm25.BM25Okapi` over all chunk texts, then combine the two
ranked lists with **reciprocal rank fusion**: `score += 1 / (60 + rank)` for each
list, then sort by the fused score. No weight tuning needed.

---

# Part D — Memory

### M1 (core 🟢) — Session window · ⏱️ ~5 min
**File:** `src/memory.py` → `SessionMemory.as_messages()`
**Goal:** return only the last `self.window` turns so follow-up questions have
context without blowing the token budget.

**Definition of done** (`python -m checks.check_memory`):
- with `window=3` and 5 turns added, returns the last 3, in order.

**Hint:** one line — a list slice (`self.turns[-self.window:]`).

### M2 (core 🟡) — Profile persistence · ⏱️ ~8 min
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

## Further ideas (if the room is flying)

- **Metadata filtering** — store each chunk's file `type` in `ingest.py` metadata,
  then use Chroma's `where=` to search only code, or only docs.
- **Result diversity (MMR)** — re-rank so the top-k aren't near-duplicates of each
  other.
- **Query expansion** — embed a couple of rephrasings of the question and merge
  their hits.

Ask if you'd like any of these scaffolded as full exercises.

---

## Where this goes next

Ingestion + embeddings + retrieval get the *right context* in front of the model;
memory carries the conversation and the person. The **RAG-generation** session
(`src/rag.py`) plugs both into a grounded, cited answer. Other layers (agent,
routing, eval) live in their own stubs — see the table in `README.md`.

---

<!-- Other presenters: add your layer's exercises below this line. -->
