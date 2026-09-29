# 🤖 Layer 4 — Agents & Orchestration

A setup-troubleshooting **agent** with tools (code search, file reading, git
blame), a **LangGraph guided onboarding flow** with mentor checkpoints, an
**MCP server** that exposes the tools inside the IDE, and **LangSmith**
observability across everything.

---

## Prerequisites

Complete these **before** the session:

1. **Layers 1–3 working** — at minimum, R1 (semantic search) must pass.
   ```bash
   python -m checks.check_retrieval    # must pass
   ```
2. **Vector index built:**
   ```bash
   python -m src.ingest                # "Indexed 30 chunks"
   ```
3. **LLM gateway configured** — `.env` has `LITELLM_PROXY_API_BASE` and
   `LITELLM_PROXY_API_KEY`:
   ```bash
   python -m checks.check_llm         # must pass
   ```
4. **LangSmith account** — sign up at [smith.langchain.com](https://smith.langchain.com/)
   (free tier). Copy your API key.
5. **Tavily API key** — sign up at [tavily.com](https://tavily.com/) (free tier).
   Copy your API key.

---

## Session setup

### 1. Add env vars to `.env`

```bash
# Add these lines to your .env in the repo root:
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-pt-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=onboarding-assistant
```

### 2. Install new packages

```bash
source .venv/bin/activate
pip install -r requirements.txt        # picks up LangChain, LangGraph, LangSmith, MCP
```

### 3. Install Jupyter (for the concept notebooks)

```bash
pip install jupyter ipykernel
```

### 4. Verify everything

```bash
python -m checks.check_llm            # LLM gateway ✅
python -m checks.check_langsmith      # LangSmith tracing ✅
jupyter notebook --version            # Jupyter installed ✅
```

---

## Concept notebooks (presenter walk-through)

The `notes/` folder (repo root) has Jupyter notebooks the presenter demos before
the exercises. Open them from the repo root:

```bash
jupyter notebook notes/                 # opens the notebook browser

# Or individually:
jupyter notebook notes/langchain.ipynb
jupyter notebook notes/langgraph.ipynb
jupyter notebook notes/langsmith.ipynb
```

| Notebook | Concepts covered |
|----------|-----------------|
| `notes/langchain.ipynb` | Chat models, prompt templates, LCEL chains (`\|` pipe), `@tool` decorator, function calling, `create_agent()` ReAct loop, Tavily search |
| `notes/langgraph.ipynb` | `StateGraph`, TypedDict state, LLM nodes, conditional edges, loops, `MemorySaver` checkpointing, `interrupt()` human-in-the-loop |
| `notes/langsmith.ipynb` | Auto-tracing, `@traceable` custom spans, agent traces, evaluation datasets, LLM-as-judge, prompt versioning (`push_prompt`/`pull_prompt`) |

> **Note:** The notebooks may have additional dependencies (e.g. `langchain-openai`).
> These are managed separately and may change — install them when the presenter
> walks through the notebooks.

---

## Exercises

Head to **[EXERCISES.md](EXERCISES.md)** for the full hands-on tasks:

| Exercise | What | File | Self-check |
|----------|------|------|------------|
| **A1** | Define 3 LangChain tools | `src/agent.py` | `python -m checks.check_agent` |
| **A2** | Build the AgentExecutor | `src/agent.py` | `python -m checks.check_agent` |
| **A3** | Run the agent | `src/agent.py` | `python -m checks.check_agent` |
| **G1** | Implement 4 graph node functions | `src/graph.py` | `python -m checks.check_graph` |
| **G2** | Routing function (conditional edges) | `src/graph.py` | `python -m checks.check_graph` |
| **G3** | Assemble + compile the StateGraph | `src/graph.py` | `python -m checks.check_graph` |
| **MCP1** | Create FastMCP server with 3 tools | `src/mcp_server.py` | `python -m checks.check_mcp` |
| **MCP2** | Add MCP resources (optional) | `src/mcp_server.py` | `python -m checks.check_mcp` |
| **LS1** | Configure and verify LangSmith | `src/langsmith_utils.py` | `python -m checks.check_langsmith` |
| **LS2** | `@traceable` on custom functions | `src/langsmith_utils.py` | `python -m checks.check_langsmith` |

Reference solutions are in `solutions/` — try first, then peek.

---

## Key concepts

| Concept | What it does | Where |
|---------|-------------|-------|
| **LangChain** | Framework for LLM apps: tools, models, prompts, agents | `src/agent.py`, `notes/langchain.ipynb` |
| **@tool** | Decorator — turns a Python function into something an LLM can call | `src/agent.py` |
| **LCEL chains** (`\|`) | Fixed pipelines: `prompt \| llm \| parser` | `notes/langchain.ipynb` |
| **AgentExecutor** | The tool-calling loop: model → tool → result → model → … → answer | `src/agent.py` |
| **LangGraph** | Stateful multi-step workflows as graphs (nodes + edges) | `src/graph.py`, `notes/langgraph.ipynb` |
| **StateGraph** | Typed state machine: `TypedDict` state, nodes (functions), edges | `src/graph.py` |
| **Conditional edges** | Routing: a function inspects state → returns next node name | `src/graph.py` |
| **Checkpointing** | `MemorySaver` persists state across calls via `thread_id` | `notes/langgraph.ipynb` |
| **Human-in-the-loop** | `interrupt()` pauses graph; `Command(resume=...)` resumes | `src/graph.py` (mentor checkpoint) |
| **MCP** | Model Context Protocol — expose tools to Cursor / Claude Code | `src/mcp_server.py` |
| **LangSmith** | Observability — traces every LLM call, tool, graph step. **Required.** | `src/langsmith_utils.py`, `notes/langsmith.ipynb` |
| **@traceable** | LangSmith decorator — adds custom functions to the trace tree | `src/langsmith_utils.py` |
| **LangSmith Eval** | Golden datasets + LLM-as-judge scoring | `notes/langsmith.ipynb` |
| **Prompt Management** | `push_prompt` / `pull_prompt` for versioning prompts | `notes/langsmith.ipynb` |

## Glossary

- **LangChain** — Python framework for composing LLM-powered applications.
- **LCEL** — LangChain Expression Language. The `|` pipe syntax for composing chains.
- **Agent** — an LLM that uses tools in a loop to take actions and answer questions.
- **Tool calling** — the LLM decides which function to call with what arguments;
  your code executes it and feeds the result back.
- **LangGraph** — library for stateful, multi-step workflows as directed graphs.
- **StateGraph** — LangGraph's core: a typed state machine (nodes + edges).
- **Checkpointing** — saving graph state so you can resume or do multi-turn.
- **Human-in-the-loop** — pausing a flow for human review (`interrupt()`).
- **MCP** — Model Context Protocol. Open standard for exposing tools to IDEs.
- **LangSmith** — observability platform for tracing LLM calls, tool use, latency,
  and cost. Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` in `.env`.
- **@traceable** — LangSmith decorator that wraps any function as a trace span.
