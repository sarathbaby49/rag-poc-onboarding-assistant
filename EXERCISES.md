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
python -m checks.check_rag           # G1 (+ G2, G3, G4); live checks need the gateway
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

# Part E — Generation · the "G" in RAG  🎤

> **Seminar layer.** Retrieval gets the *right context* in front of the model.
> Generation turns it into a **grounded, cited answer** — and, just as important,
> knows when to say **"I don't know."** This is the payoff the whole repo was
> building toward.

**File you edit:** `src/rag.py`. **Self-check:** `python -m checks.check_rag`.
Difficulty: 🟢 easy · 🟡 medium · 🔵 explore.

### Before you start
- This layer calls a language model through the **LiteLLM gateway**, so the *live*
  exercises need `.env` (copy `.env.example`) — verify once with
  `python -m checks.check_llm`. **But the safety behaviour is key-free:** G2's
  abstention and G3's citation filter self-check with **no API key at all**.
- **Generation depends on Retrieval.** Implement **R1** (`semantic_search`) first,
  or copy `solutions/retrieve.py`. G2 also needs **R3** (`confident_hits`); G4
  needs **M1** (`SessionMemory.as_messages`).
- **Handed to you** in `src/rag.py`: `SYSTEM_PROMPT` (the assistant's
  "constitution") and `format_context` (numbers the chunks so the model can cite
  `[1]`, `[2]`) — the scaffolding, just like `_encoder()`/`_collection()` were in
  the retrieval exercises.

**Watch it live — two UIs:**
- `streamlit run rag_app.py` — the **RAG Exercise Lab**. A sidebar selector picks
  which function to test (**G1** `answer`, **G2** `answer_or_abstain`, **G4**
  `answer_conversational`), so you can work the exercises in any order. Because it
  goes through `src/rag_helper.py`, an exercise (or a dependency like R1/R3/M1)
  you haven't finished yet shows a friendly "implement this next" message instead
  of crashing — the page doubles as a live progress board.
- `streamlit run app.py` — the polished **end-user chat demo**. It calls
  `answer()` directly (no safety net), so it needs G1 done, and is what you'd show
  off once the layer works.

---

## ⏱️ Run of show (~45 min)

| Time | What |
|------|------|
| 0:00–0:10 | Concept: retrieval → grounded prompt → cited answer. Why "answer only from context" is the whole game. Demo `app.py` on the reference build. |
| 0:10–0:24 | **G1** — `answer` (everyone) · self-check · watch it live in `rag_app.py` |
| 0:24–0:36 | Pick your depth: **G2** abstention (the safety property) · **G3** trustworthy citations |
| 0:36–0:44 | **G4** conversational memory · or **G5** prompt-craft the constitution |
| 0:44–0:45 | Recap: how ingestion + embeddings + retrieval + memory all fed this one answer. |

Fast finishers: **G4** (memory) or the **Further ideas** below. Nobody needs to finish everything.

---

### G1 (core 🟢) — Grounded, cited generation · ⏱️ ~12 min
**File:** `src/rag.py` → `answer()`
**Goal:** retrieve the top-k chunks, hand them to the model with the `SYSTEM_PROMPT`,
and make it answer **only** from that context, citing `[n]`. Return
`{"answer", "sources"}`.

**Definition of done** (`python -m checks.check_rag`, live check needs the gateway):
- returns a dict with `"answer"` (str) and `"sources"` (the retrieved hits)
- asking *"How do I set up my local environment?"* returns `setup.md` among the
  sources and an answer containing a `[n]` citation marker

**Hints:**
- `hits = semantic_search(question, k)` → `context = format_context(hits)`
- `complete([{system}, {user}], max_tokens=config.MAX_TOKENS)` where the user
  content is `f"Context sources:\n\n{context}\n\nQuestion: {question}"`
- Reference: `solutions/rag.py`.

**The key idea to say out loud:** the model never sees the corpus — only the 4
chunks retrieval chose, plus the rules. That's why the answer is checkable.

### G2 (core 🟡) — Honest "I don't know" (abstention) · ⏱️ ~10 min  · *(needs R1 + R3)*
**File:** `src/rag.py` → `answer_or_abstain()`
**Goal:** `semantic_search` **always** returns k chunks — even for a question the
corpus can't answer — so a naive assistant answers from junk. Gate on confidence:
if nothing clears the bar, hand off to a human **without calling the model** (no
tokens spent, zero chance of a hallucinated answer).

**Definition of done** (`python -m checks.check_rag` — the abstain path is **key-free**):
- at a high `min_score` (e.g. `0.99`) it returns `ABSTAIN_MESSAGE` and **empty
  sources**, and never reaches the gateway
- on-topic (with the gateway) it returns a real cited answer

**Hint:** `hits = confident_hits(question, k, min_score)` (R3); if `not hits`,
`return {"answer": ABSTAIN_MESSAGE, "sources": []}`; otherwise generate like G1.
`ABSTAIN_MESSAGE` is already defined for you.

### G3 (stretch 🔵) — Trustworthy citations · ⏱️ ~8 min
**File:** `src/rag.py` → `used_sources()`
**Goal:** grounding you don't verify is just a promise. You give the model k
sources but it may only cite `[2]` — so the **Sources** panel should show `[2]`,
not all k. Pure function, **no key needed**.

**Definition of done** (`python -m checks.check_rag`, key-free):
- `"See [1] and [3]."` over 3 hits → returns hits 1 and 3 (in order)
- text with no `[n]` markers → returns `[]`

**Hint:** `re.findall(r"\[(\d+)\]", answer_text)` gives the cited numbers; one
comprehension filters the hits by 1-based position. Then wire it into
`answer()` so `sources = used_sources(text, hits)`.

### G4 (stretch 🟡) — Conversational RAG (memory) · ⏱️ ~10 min  · *(needs M1)*
**File:** `src/rag.py` → `answer_conversational()`
**Goal:** *"how do I run it?"* → *"what about the tests?"* only works if the model
sees the last few turns. Splice the session **window** into the call.

**Definition of done** (`python -m checks.check_rag`, live check needs the gateway):
- returns an answer and records **both** turns (memory grows by 2)
- a follow-up that omits the subject still answers on-topic

**Hint:** `messages = [{system}, *memory.as_messages(), {user}]` (M1's window keeps
the token budget bounded); after generating, `memory.add("user", q)` and
`memory.add("assistant", text)`.

### G5 (explore 🔵) — Prompt-craft the "constitution" · ⏱️ ~8 min
**File:** `src/rag.py` → `SYSTEM_PROMPT` *(no self-check — you're the judge)*
The `SYSTEM_PROMPT` is the single line that makes RAG **safe**. Experiment and
re-run a few questions in `rag_app.py` (or `app.py`) after each change:
- **Remove** *"Answer ONLY using the numbered context sources"* — watch it drift
  back to its own memory and answer things the docs never said.
- **Add** *"Quote the exact line you're citing."* — do citations get sharper?
- **Retune the tone** for a nervous first-day joiner.

**What to observe:** how much answer quality and faithfulness ride on one prompt.

---

## Further ideas (if the room is flying)
- **Streaming** — stream tokens to the UI instead of waiting for the whole answer.
- **Query condensing** (better G4) — before retrieving, rewrite the follow-up into
  a standalone question using the history, so retrieval isn't confused by pronouns.
- **Faithfulness eval** — score each answer with the golden set in `src/eval/`
  (keyword coverage, or LLM-as-judge). Ties this layer to Layer 6.

## Where this fits
G1–G4 are the capstone: **ingestion + embeddings + retrieval + memory all exist to
feed this one grounded, cited answer.** It's the "G" the "R" was built for — and the
handoff point to the agent, routing and eval layers (see the table in `README.md`).

---

<!-- Other presenters: add your layer's exercises below this line. -->
