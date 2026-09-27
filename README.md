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
python -m src.ingest          # builds the local vector index → "Indexed 12 chunks"
```

Then head to **[EXERCISES.md](EXERCISES.md)** for the hands-on tasks.

> **Retrieval + Memory need no API key** (local embeddings + plain Python).
> The RAG-generation layer (`src/rag.py`) calls models through the **LiteLLM
> gateway** — no direct Claude/OpenAI key, just the proxy URL + key in `.env`
> (see `.env.example`). Check it with `python -m checks.check_llm`.

---

## What's in the box

| Status | Piece | File |
|--------|-------|------|
| ✅ Works | Ingestion: load → chunk → embed → store | `src/ingest.py` |
| ✅ Works | The dials (chunk size, top-k, models) | `src/config.py` |
| ✅ Works | Retrieval intuition explorer (no key) | `explore.py` |
| 📝 **Exercise** | **Retrieval** — semantic + hybrid search | `src/retrieve.py` |
| 📝 **Exercise** | **Memory** — session window + profile persistence | `src/memory.py` |
| 🔒 Other session | RAG generation (retrieve → grounded, cited answer) | `src/rag.py`, `app.py` |
| 🔒 Later layers | Agent, guided flow, MCP, routing, eval | `src/agent.py`, `graph.py`, `mcp_server.py`, `models.py`, `eval/` |

- **✅ Works** — runs today; don't edit, build against it.
- **📝 Exercise** — a stub that raises `NotImplementedError`; you implement it and
  self-check with `python -m checks.check_*`.
- **🔒** — owned by another session/presenter; stubbed on purpose.

## Self-check your work

```bash
python -m checks.check_retrieval     # R1 (required) + R2 (stretch)
python -m checks.check_memory        # M1 + M2
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
├── app.py                    # 🔒 Streamlit UI (works once src/rag.py is built)
├── data/sample_company/      # fake team knowledge (docs, code, tickets, slack)
├── checks/                   # self-check scripts (pass/fail)
│   ├── check_retrieval.py
│   └── check_memory.py
├── solutions/                # reference answers — try first!
│   ├── retrieve.py
│   └── memory.py
└── src/
    ├── config.py             # ✅ all the dials
    ├── ingest.py             # ✅ builds the index
    ├── retrieve.py           # 📝 EXERCISE (R1, R2)
    ├── memory.py             # 📝 EXERCISE (M1, M2)
    ├── rag.py                # 🔒 RAG-generation session
    ├── agent.py / graph.py / mcp_server.py / models.py   # 🔒 later layers
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
