# 🧭 Engineering Onboarding Assistant — Upskill Skeleton

A hands-on teaching repo for building a RAG onboarding assistant, layer by layer.
It answers new-joiner questions from a fake company's docs, code, tickets and
Slack — grounded with citations.

This is a **baseline skeleton**: a few pieces work out of the box so you have
something to build against, and the rest are **stubbed exercises** that each
session's presenter and participants fill in.

---

## Quickstart

> **Python 3.10+ required.**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.ingest          # builds the local vector index → "Indexed 30 chunks"
```

Then head to **[EXERCISES.md](EXERCISES.md)** for the hands-on tasks.

**Prefer a UI?** Launch the **Retrieval Lab** — a chat-style playground to query the
vector DB, tune ingestion settings and re-ingest with a click, and see the store as
a 2D map (no API key):

```bash
streamlit run retrieval_lab.py
```

Working the **generation** exercises (G1–G4)? The **RAG Exercise Lab** is the
matching UI — pick which function to test in the sidebar and chat against it.
Because it goes through `src/rag_helper.py`, an exercise you haven't finished yet
shows a friendly "implement this next" message instead of crashing the page:

```bash
streamlit run rag_app.py
```

Working the **agents & orchestration** exercises (A1–A3, G1–G3, MCP1–2, LS1–2)?
The **Orchestration Lab** is the matching UI — chat with the agent, run the
onboarding graph interactively, test MCP tools, and verify LangSmith config:

```bash
streamlit run orchestration_lab.py
```

Working the **model selection & cost** exercises (C1–C4)? The **Cost Lab**
compares models side by side, tests your router and per-step graph models,
shows prompt-cache hits, and keeps a running (projected) bill:

```bash
streamlit run cost_lab.py
```

> **Retrieval + Memory need no API key** (local embeddings + plain Python).
> The RAG-generation layer (`src/rag.py`) calls models through the **LiteLLM
> gateway** — no direct Claude/OpenAI key, just the proxy URL + key in `.env`
> (see `.env.example`). Check it with `python -m checks.check_llm`.

---

## What's in the box

| Status | Piece | File |
|--------|-------|------|
| ✅ Works | Ingestion pipeline (boundary-aware chunking) + the dials | `src/ingest.py`, `src/config.py` |
| ✅ Works | Retrieval intuition explorer (no key) | `explore.py` |
| ✅ Works | **Retrieval Lab** — chat UI + ingestion controls + vector map | `retrieval_lab.py` |
| ✅ Works | **RAG Exercise Lab** — chat UI for G1–G4; unfinished exercises show a TODO instead of crashing | `rag_app.py`, `src/rag_helper.py` |
| 📝 **Exercise** | **Ingestion** — compare chunkers (I1, optional), bring your own data (I2) | `src/ingest.py`, `data/` |
| 📝 **Exercise** | **Embeddings** — cosine by hand (E1), swap model (E2) | `src/embeddings.py`, `src/config.py` |
| 📝 **Exercise** | **Retrieval** — semantic (R1), hybrid (R2), confidence (R3) | `src/retrieve.py` |
| 📝 **Exercise** | **Memory** — session window (M1), semantic recall (M4); `JoineeProfile` provided | `src/memory.py` |
| 🔒 Other session | RAG generation (retrieve → grounded, cited answer) | `src/rag.py`, `app.py` |
| 📝 **Exercise** | **Memory** — session window + profile persistence | `src/memory.py` |
| 📝 **Exercise** | **Generation** — grounded, cited answer (G1), abstain (G2), citations (G3), memory (G4) | `src/rag.py`, `app.py` |
|  📝 **Exercise** | **Agents & Orchestration** (LangChain, LangGraph, LangSmith, MCP)         | `[layer4-agents-and-orchestration/](layer4-agents-and-orchestration/)` |
| 📝 **Exercise** | **Model Selection & Cost** — cost meter (C1), router (C2), model per graph step (C3), prompt caching (C4) | [`layer5-model-selection-and-cost/`](layer5-model-selection-and-cost/) |
| 🔒 Later layers | Evaluation | `eval/` |

- **✅ Works** — runs today; don't edit, build against it.
- **📝 Exercise** — a stub that raises `NotImplementedError`; you implement it and
  self-check with `python -m checks.check_*`.
- **🔒** — owned by another session/presenter; stubbed on purpose.

## Self-check your work

```bash
python -m checks.check_ingest        # I1 (boundary-aware chunking)
python -m checks.check_embeddings    # E1 (cosine similarity)
python -m checks.check_retrieval     # R1 (required) + R2, R3 (reported)
python -m checks.check_memory        # M1 + M2
python -m checks.check_rag           # G1 + G2, G3, G4 (live checks need the gateway)
python -m checks.check_agent         # A1 (tools) + A2 (agent) + A3 (run)
python -m checks.check_graph         # G1 (nodes) + G2 (routing) + G3 (graph)
python -m checks.check_mcp           # MCP1 (server + tools) + MCP2 (resources)
python -m checks.check_langsmith     # LS1 (config) + LS2 (@traceable)
```

Reference answers live in **`solutions/`** — try the exercise first, then peek if
you're stuck or out of time.

---

## How to run the session (for presenters)

- **One branch (`main`) for everyone.** Participants clone `main` and edit the
  exercise files locally. No branch switching during the session.
- **Solutions live in `solutions/`** (committed in `main`) so anyone can unblock.
- Building the repo as a presenter? Use a feature branch per layer while you
  develop, then merge to `main`. That's a dev-workflow detail — participants never
  see it.

## Repo layout

```
onboarding-assistant-rag/
├── README.md                 # you are here
├── EXERCISES.md              # the hands-on tasks + run-of-show
├── explore.py                # ✅ retrieval intuition tool (no key)
├── retrieval_lab.py          # ✅ Streamlit lab for the R exercises
├── rag_app.py                # ✅ Streamlit lab for the G exercises (crash-safe)
├── orchestration_lab.py       # ✅ Streamlit lab for agent + graph + MCP + LangSmith exercises
├── cost_lab.py               # ✅ Streamlit lab for the Layer 5 cost & routing exercises
├── app.py                    # 🔒 Streamlit end-user demo (works once src/rag.py is built)
├── data/sample_company/      # Acme Shop corpus: docs (payments, auth, setup),
│                             #   code, a Jira export & Slack history (~30 chunks)
├── layer4-agents-and-orchestration/  # 📦 separate session
│   ├── README.md             #   session setup, prerequisites, concepts
│   └── EXERCISES.md          #   hands-on exercises (A1–3, G1–3, MCP1–2, LS1–2)
├── layer5-model-selection-and-cost/  # 📦 separate session
│   ├── README.md             #   prerequisites, files
│   └── EXERCISES.md          #   hands-on exercises (C1–C4)
├── notes/                    # 📓 concept notebooks (Layer 4 presenter walk-through)
│   ├── langchain.ipynb
│   ├── langgraph.ipynb
│   └── langsmith.ipynb
├── checks/                   # self-check scripts (pass/fail)
│   ├── check_retrieval.py
│   ├── check_memory.py
│   ├── check_agent.py        # ← Layer 4
│   ├── check_graph.py        # ← Layer 4
│   ├── check_mcp.py          # ← Layer 4
│   ├── check_langsmith.py    # ← Layer 4
│   ├── check_cost.py         # ← Layer 5
│   ├── check_routing.py      # ← Layer 5
│   └── check_caching.py      # ← Layer 5
├── solutions/                # reference answers — try first!
│   ├── retrieve.py
│   ├── memory.py
│   ├── agent.py              # ← Layer 4
│   ├── graph.py              # ← Layer 4
│   ├── mcp_server.py         # ← Layer 4
│   ├── langsmith_utils.py    # ← Layer 4
│   ├── cost.py               # ← Layer 5
│   ├── models.py             # ← Layer 5
│   └── caching.py            # ← Layer 5
└── src/
    ├── config.py             # ✅ all the dials
    ├── ingest.py             # ✅ builds the index
    ├── retrieve.py           # 📝 EXERCISE (R1, R2)
    ├── memory.py             # 📝 EXERCISE (M1, M2)
    ├── rag.py                # 📝 EXERCISE (G1–G4) — the "G" in RAG
    ├── rag_helper.py         # ✅ crash-safe wrappers used by rag_app.py
    ├── agent.py              # 📝 EXERCISE (A1, A2, A3) — Layer 4
    ├── graph.py              # 📝 EXERCISE (G1, G2, G3) — Layer 4
    ├── mcp_server.py         # 📝 EXERCISE (MCP1, MCP2) — Layer 4
    ├── langsmith_utils.py    # 📝 EXERCISE (LS1, LS2) — Layer 4
    ├── cost.py               # 📝 EXERCISE (C1) — Layer 5
    ├── models.py             # 📝 EXERCISE (C2, C3) — Layer 5
    ├── caching.py            # 📝 EXERCISE (C4) — Layer 5
    ├── cost_helper.py        # ✅ Layer 5 plumbing used by cost_lab.py
    └── eval/                 # 🔒 evaluation layer
```

## Glossary

- **RAG** — Retrieval-Augmented Generation. Fetch relevant text, then let the
  model answer *using that text* instead of its own memory.
- **Embedding** — a vector representation of text; similar meaning ⇒ nearby vectors.
- **Chunk** — a small slice of a document; retrieval works on chunks, not files.
- **Vector store** — a database that finds the nearest vectors fast (here: Chroma).
- **Top-k** — how many chunks we retrieve and give the model.
- **Hybrid search** — blending keyword (BM25) and semantic ranking.
- **Grounding / citation** — tying each claim to a source so answers are checkable.
