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
python -m checks.check_memory        # M1, M3, M4
python -m checks.check_ingest        # regression check for the default chunker
```

---

## ⏱️ Run of show (~50 min)

| Time | What |
|------|------|
| 0:00–0:10 | Concept: embeddings, cosine, ingest → store → retrieve. Live-drive `explore.py`. |
| 0:10–0:22 | **R1** — `semantic_search` (everyone) · self-check |
| 0:22–0:38 | Pick your depth: **E1** cosine · **I1** compare chunkers · **I2** bring your own data · **R3** confidence · **E2** swap model |
| 0:38–0:46 | Memory: **M1** session window · **M3** summary buffer · **M4** semantic recall (`JoineeProfile` is provided) |
| 0:46–0:50 | Recap: short-term vs long-term, why bigger context isn't the fix, how it feeds the RAG-generation layer next. |

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

> **Three assignments here — M1, M3 and M4.** `JoineeProfile` is **provided,
> already working** (read it) as the worked example of persistent long-term
> memory. The rest — the session window (M1), the summary buffer (M3) and
> semantic recall (M4) — are yours to implement.

### The two topics in one minute

**1. Short-term vs long-term memory** — two different jobs. Short-term (**M1**) is
the **live conversation**: the last few turns, cheap, gone when the chat ends.
Long-term is the **durable store**: who the person is — either a saved profile
(`JoineeProfile`, provided) or memories you search by meaning (**M4**). A real app
runs both.

**2. When context length alone isn't enough** — "just use a bigger window / a
longer-context model" fails three ways: it's **expensive** (you pay for every
token, every turn), it's **slow**, and quality drops as the real answer is
**buried** among thousands of irrelevant tokens ("lost in the middle"). The
durable fixes are to **compress** old turns into a running summary (**M3**) and to
**retrieve** only the relevant pieces on demand (**M4**, which is just R1's
retrieval pointed at the conversation instead of docs).

> 📖 Deeper write-up + diagrams: **`docs/layer3-memory-and-context.md`**.

### M1 (core 🟢) — Session window · ⏱️ ~5 min
**File:** `src/memory.py` → `SessionMemory.as_messages()`
**Goal:** return only the last `self.window` turns so follow-up questions have
context without blowing the token budget.

**Definition of done** (`python -m checks.check_memory`):
- with `window=3` and 5 turns added, returns the last 3, in order.

**Hint:** one line — a list slice (`self.turns[-self.window:]`).

### M3 (core 🟡) — Summary buffer (compress overflow) · ⏱️ ~8 min
**File:** `src/memory.py` → `SummaryBufferMemory.add()`
**Goal:** keep the last `self.window` turns verbatim, but instead of *dropping*
older turns like a plain window does, **fold them into a running summary** so the
gist of the whole conversation survives at a fixed token cost. `_extractive_summary`
(the summariser) and `as_context()` (assembles summary + live turns) are provided —
you write the overflow logic in `add()`.

**Definition of done** (`python -m checks.check_memory`):
- with `window=2` and 4 turns added, `turns` holds only the last 2 (in order) and
  `summary` contains the two oldest turns that scrolled off.

**Hint:** append the turn, then `while len(self.turns) > self.window:` pop the
oldest (`self.turns.pop(0)`) and roll it in —
`self.summary = self.summarize(self.summary, old["role"], old["content"])`.

### M4 (core 🟡) — Semantic recall (memory as retrieval) · ⏱️ ~8 min
**File:** `src/memory.py` → `MemoryStore.recall()`  · *(reuses `embed`)*
**Goal:** recall the *relevant* memory, not the *recent* one. A window can only
return the latest turns; but the fact you need ("I'm on the payments team") may be
50 turns back. Embed every memory once, then at query time return the closest ones
by cosine — this is R1's retrieval aimed at the conversation. (The best "aha" of
the session, and it reuses the embeddings + cosine you already built.)

**Definition of done** (`python -m checks.check_memory`):
- after remembering three facts (payments team, coffee machine, standup time),
  `recall("which team am I part of?", k=1)` returns the **payments** memory —
  even though it was added *first* and the standup memory is more recent.

**Hint:** `from src.embeddings import embed`; embeddings are normalised, so cosine
is just the dot product — `sorted(items, key=lambda it: sum(a*b for a,b in zip(q, it["embedding"])), reverse=True)[:k]`.

### Provided (read, don't code) — `JoineeProfile`
Long-term memory of a joinee's **role + onboarding progress**, persisted to a JSON
file (`.profiles/<name>.json`) so it survives a restart. It's the worked example of
durable long-term memory — `record_step()` auto-saves progress. Read it in
`src/memory.py`; nothing to implement.

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
memory carries the conversation and the person — short-term for the live chat,
long-term for who they are, and semantic recall for when a window or a bigger
context isn't enough. The **RAG-generation** session
(`src/rag.py`) plugs both into a grounded, cited answer. Other layers (agent,
routing, eval) live in their own stubs — see the table in `README.md`.

---

<!-- Other presenters: add your layer's exercises below this line. -->
