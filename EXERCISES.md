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
python -m src.ingest          # builds the vector index → "Indexed 30 chunks"
```

The sample company is **Acme Shop**, an e-commerce backend — the corpus has docs
(payments, authentication, setup, README), service code (`payment_providers.py`,
`auth.py`), a Jira export and Slack history (~30 chunks).

Get a feel for retrieval first (no code yet):

```bash
python explore.py stats                                  # what's in the store
python explore.py query "what payment providers do we use?"   # semantic search, with scores
python explore.py query "how is authentication set up?"       # finds authentication.md
python explore.py compare "log in" "authenticate a user"
python explore.py keyword "RazorpayProvider"             # keyword vs semantic → why hybrid
```

**Files you edit:** `src/embeddings.py`, `src/retrieve.py`, `src/memory.py` (and
`data/` for I2). **Self-check any exercise** with its command below (instant
✅/❌). Stuck or out of time? Answers are in **`solutions/`** — try first, then peek.

```bash
python -m checks.check_embeddings    # E1
python -m checks.check_retrieval     # R1 (+ R2, R3 reported)
python -m checks.check_memory        # M1, M2
python -m checks.check_ingest        # regression check for the default chunker
```

---

## ⏱️ Run of show (~50 min)

| Time | What |
|------|------|
| 0:00–0:10 | Concept: embeddings, cosine, ingest → store → retrieve. Live-drive `explore.py`. |
| 0:10–0:22 | **R1** — `semantic_search` (everyone) · self-check |
| 0:22–0:38 | Pick your depth: **E1** cosine · **I1** compare chunkers · **I2** bring your own data · **R3** confidence · **E2** swap model |
| 0:38–0:48 | Memory: **M1** session window · **M2** profile persistence |
| 0:48–0:50 | Recap: how it all feeds the RAG-generation layer next. |

Fast finishers go for **R2** (hybrid search). Nobody needs to finish everything.

---

# Part A — Ingestion

### I1 (explore 🔵) — Compare chunking strategies · ⏱️ ~8 min
Chunking is the single biggest retrieval-quality lever, so the pipeline **ships
with boundary-aware chunking** (`chunk_text_smart` in `src/ingest.py`) as the
default — it ends each chunk at a whitespace/newline so words stay whole. The
naive fixed-size `chunk_text` is kept as a baseline to compare against.

**Explore (no required code):**
1. In the **Retrieval Lab**, tick **"Use naive fixed-size chunking (to compare)"**
   in the sidebar and click **🔁 Re-ingest data**.
2. Query something and watch the chunks (hover `chars` on the map) and the scores
   shift; fixed-size can cut words mid-token and split ideas.
3. Untick it and re-ingest to return to the default boundary-aware chunker.

**Stretch (code):** open `src/ingest.py` and write your own smarter splitter —
sentence-aware, or Markdown-heading / code-function aware — and make `build_index`
use it. (`python -m checks.check_ingest` guards that the default chunker keeps
words whole.)

### I2 (explore 🔵) — Bring your own data · ⏱️ ~10 min
**Files:** add a new file under `data/sample_company/` (any `.md`, `.txt`, `.py`,
`.json`, or `.jsonl` — those are the types ingest reads).
**Goal:** run the whole ingest → retrieve loop on data *you* wrote, and get a feel
for what makes retrieval work well (or not).

**Steps:**
1. Create a document with a few facts only *you* know — e.g.
   `data/sample_company/my_notes.md` with a couple of short Q&A-style paragraphs
   (a mini runbook, a made-up policy, your team's quirks).
2. Re-ingest so it's embedded and stored:
   - CLI: `python -m src.ingest`, **or**
   - click **🔁 Re-ingest data** in the Retrieval Lab (`streamlit run retrieval_lab.py`).
3. Query for something only your file answers — in the Lab's chat box or
   `python explore.py query "…"`.

**What "done" looks like** (no automated check — you're the judge):
- A question phrased *like your text* retrieves your file at rank 1–2 with a
  healthy score.
- Then try a **paraphrase** that shares no exact words. Does it still surface? If
  not, that's the chunking/wording lesson — tighten the doc or adjust chunk size.

**Things to try:**
- Add a long file and watch it become several chunks (the count grows, and a new
  cluster appears on the **Vector map**).
- Add a code file with a distinctive symbol, then compare `explore.py query` vs
  `explore.py keyword` on that symbol (motivates hybrid search, R2).
- Ask something your data does *not* cover and watch the scores drop — the
  motivation for the R3 confidence threshold.

**Tidy up:** delete your file and re-ingest to return to the standard corpus (or
keep it — your call).

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

**See YOUR code working:** the automated check is `python -m checks.check_retrieval`.
To watch it live, open the **Retrieval Lab** (`streamlit run retrieval_lab.py`) and set
the sidebar **Backend** to *My semantic_search (R1)* — the chat and the vector map now
run on **your** function. (Before it's implemented, that backend shows a reminder;
switch to *Built-in* to compare against the reference.)

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
(like `RazorpayProvider`) *and* paraphrases both rank well.

**Definition of done:** `RazorpayProvider` surfaces `code/payment_providers.py` in
the top-k (plain semantic often misses an exact class name — that's the motivation).

**Hint:** build a `rank_bm25.BM25Okapi` over all chunk texts, then combine the two
ranked lists with **reciprocal rank fusion**: `score += 1 / (60 + rank)` for each
list, then sort by the fused score. No weight tuning needed.

**See it live:** in the Retrieval Lab, flip the **Backend** between *My semantic_search
(R1)* and *My hybrid_search (R2)* and query an exact code symbol (e.g. `RazorpayProvider`)
— your hybrid version should rank the code file higher.

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

# Part E — Agents & Orchestration (Layer 4)

**This is a separate session with its own setup and prerequisites.**

👉 Head to **[`layer4-agents-and-orchestration/`](layer4-agents-and-orchestration/)** for the
full README, setup instructions, and exercises (A1–3, G1–3, MCP1–2, LS1–2).

---

## Further ideas (if time permits)

- **Metadata filtering** — store each chunk's file `type` in `ingest.py` metadata,
  then use Chroma's `where=` to search only code, or only docs.
- **Result diversity (MMR)** — re-rank so the top-k aren't near-duplicates of each
  other.
- **Query expansion** — embed a couple of rephrasings of the question and merge
  their hits.
- **Agent memory** — give the agent `SessionMemory` so follow-up questions work
  across tool calls.
- **Multi-step graph** — extend the LangGraph flow with a "first ticket" node that
  assigns a starter task and a "first PR" node that reviews their code.
- **LangSmith evaluation** — use LangSmith's evaluation framework to build a
  golden-set test for agent quality (latency, tool usage, answer correctness).

---

## Where this goes next

Ingestion + embeddings + retrieval get the *right context* in front of the model;
memory carries the conversation and the person. The **RAG-generation** session
(`src/rag.py`) plugs both into a grounded, cited answer. The **agent layer**
(Layer 4) adds tool use and guided flows, and **MCP** makes it all available
inside your IDE. Other layers (routing, eval) live in their own stubs — see the
table in `README.md`.

---

<!-- Other presenters: add your layer's exercises below this line. -->
