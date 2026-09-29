# 🤖 Exercises — Agents & Orchestration (Layer 4)

Hands-on tasks for the Layer 4 session. You build a **setup-troubleshooting
agent** (LangChain), a **guided onboarding flow** (LangGraph), an **MCP server**
for IDE integration, and wire up **LangSmith** observability across everything.

Difficulty: 🟢 easy · 🟡 medium · 🔵 explore.

---

## Before you start

Make sure you've completed the [session setup](README.md#session-setup):

- [ ] `.env` has LangSmith vars (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, etc.)
- [ ] `.env` has `TAVILY_API_KEY`
- [ ] `pip install -r requirements.txt` done (from repo root)
- [ ] `python -m checks.check_llm` passes
- [ ] Jupyter installed for the concept notebooks

**Files you edit:** `src/agent.py`, `src/graph.py`, `src/mcp_server.py`,
`src/langsmith_utils.py` (all in the repo root `src/` folder).

**Self-check:**

```bash
python -m checks.check_agent         # A1 (tools) + A2 (agent) + A3 (run)
python -m checks.check_graph         # G1 (nodes) + G2 (routing) + G3 (graph)
python -m checks.check_mcp           # MCP1 (server + tools) + MCP2 (resources)
python -m checks.check_langsmith     # LS1 (config) + LS2 (@traceable)
```

Stuck or out of time? Reference answers are in `solutions/` — try first,
then peek.

---

## ⏱️ Run of show (~50 min)

| Time | What |
|------|------|
| 0:00–0:08 | Concept: agents vs chatbots, tool calling, LangChain tools. |
| 0:08–0:18 | **A1** — define the 3 tools · **A2** — build the agent · self-check |
| 0:18–0:21 | **A3** — run the agent · try different questions |
| 0:21–0:28 | **G1** — graph node functions · **G2** — routing · self-check |
| 0:28–0:35 | **G3** — assemble the graph · run the flow |
| 0:35–0:42 | **MCP1** — expose tools as MCP server · connect to Cursor |
| 0:42–0:47 | **LS1** — configure LangSmith · **look at traces in the dashboard** |
| 0:47–0:50 | **LS2** — @traceable on custom functions · recap |

Fast finishers go for **MCP2** (resources). LangSmith comes last so you have
agent + graph traces to explore in the dashboard.

---

## LangChain Agent

### A1 (core 🟢) — Define LangChain tools · ⏱️ ~8 min

**File:** `src/agent.py` → `code_search()`, `read_file()`, `git_blame()`
**Goal:** wrap the three setup-troubleshooting tools using LangChain's `@tool`
decorator so an LLM can discover and call them.

**What you'll use:**

- `from langchain_core.tools import tool` — the `@tool` decorator
- `semantic_search(query, k=3)` from `src.retrieve` — for code_search
- `subprocess.run(["git", "blame", ...])` — for git_blame
- Path resolution + sandboxing — for read_file

**Definition of done** (`python -m checks.check_agent`, section A1):

- All three functions are callable
- They're decorated with `@tool` (have a `.name` attribute)
- `read_file` rejects paths outside the repo root

**Hints:**

```python
from langchain_core.tools import tool

@tool
def code_search(query: str) -> str:
    """Search the codebase and docs for a keyword or concept."""
    hits = semantic_search(query, k=3)
    return "\n\n".join(f"{h['source']}:\n{h['text']}" for h in hits)
```

### A2 (core 🟡) — Build the agent · ⏱️ ~8 min

**File:** `src/agent.py` → `build_agent()`
**Goal:** create an `AgentExecutor` that binds the three tools to a chat model.

**What you'll use:**

- `ChatOpenAI` from `langchain_openai` — talks to the LiteLLM proxy
- `create_tool_calling_agent` from `langchain.agents` — creates the agent
- `AgentExecutor` from `langchain.agents` — runs the tool-calling loop
- `ChatPromptTemplate` + `MessagesPlaceholder` for the prompt

**Definition of done** (`python -m checks.check_agent`, section A2):

- `build_agent()` returns an object with `.invoke()` and `.tools`
- At least 3 tools are bound

**Hints:**

```python
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url=config.LITELLM_PROXY_API_BASE or None,
    api_key=config.LITELLM_PROXY_API_KEY or None,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
return AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
```

### A3 (core 🟢) — Run the agent · ⏱️ ~3 min

**File:** `src/agent.py` → `run_agent()`
**Goal:** invoke the agent on a question and return the answer string.

**Definition of done** (`python -m checks.check_agent`, section A3):

- `run_agent("What files are in this repo?")` returns a non-empty string

**Hint:** `build_agent().invoke({"input": question})["output"]`

**Try it:**

```bash
python -m src.agent "Where is the payment provider code and who last changed it?"
```

---

## LangGraph Flow

### G1 (core 🟡) — Implement graph node functions · ⏱️ ~10 min

**File:** `src/graph.py` → `welcome()`, `setup_help()`, `mentor_checkpoint()`, `architecture_tour()`
**Goal:** implement the four nodes of the onboarding state machine. Each node
receives the current `OnboardingState` and returns a partial update dict.

**Key idea:** LangGraph nodes are plain functions. They take state in, return
partial updates out. LangGraph merges the update into the running state.

**Definition of done** (`python -m checks.check_graph`, section G1):

- `welcome` sets `stage="setup"` and adds "welcome" to `completed_steps`
- `setup_help` adds an answer message; sets `needs_mentor=True` if the user's
  message contains "error", "fail", "stuck", "broken", etc.
- `mentor_checkpoint` clears `needs_mentor`, advances to `stage="architecture"`
- `architecture_tour` sets `stage="first_ticket"`

**Hints:**

```python
def welcome(state: OnboardingState) -> dict:
    name = state["joinee"]
    return {
        "stage": "setup",
        "messages": state["messages"] + [{"role": "assistant", "content": f"Welcome, {name}!"}],
        "completed_steps": state["completed_steps"] + ["welcome"],
    }
```

### G2 (core 🟢) — Routing function · ⏱️ ~3 min

**File:** `src/graph.py` → `route_after_setup()`
**Goal:** decide whether to go to the mentor checkpoint or skip to architecture.

**Definition of done** (`python -m checks.check_graph`, section G2):

- Returns `"checkpoint"` when `needs_mentor` is True
- Returns `"architecture"` when `needs_mentor` is False

**Hint:** one line — `return "checkpoint" if state["needs_mentor"] else "architecture"`

### G3 (core 🟡) — Assemble the graph · ⏱️ ~8 min

**File:** `src/graph.py` → `build_graph()`
**Goal:** wire the nodes and edges into a `StateGraph`, compile it, and return.

**What you'll use:**

- `from langgraph.graph import StateGraph, END`
- `StateGraph(OnboardingState)` — creates the graph
- `g.add_node(name, function)` — adds a node
- `g.set_entry_point(name)` — sets the start node
- `g.add_edge(from, to)` — adds a fixed edge
- `g.add_conditional_edges(from, routing_fn)` — adds a conditional edge
- `g.compile(interrupt_before=[...])` — compiles the graph

**Definition of done** (`python -m checks.check_graph`, section G3):

- `build_graph()` returns a compiled graph with `.invoke()`
- The happy path (no mentor needed) reaches `stage="first_ticket"`
- `welcome` and `architecture` appear in `completed_steps`

**The graph shape:**

```
welcome ──► setup ──┬──► architecture ──► END
                    │
                    └──► checkpoint ──► architecture
                    (only if needs_mentor=True)
```

**Hints:**

```python
from langgraph.graph import StateGraph, END

g = StateGraph(OnboardingState)
g.add_node("welcome", welcome)
g.add_node("setup", setup_help)
g.add_node("checkpoint", mentor_checkpoint)
g.add_node("architecture", architecture_tour)

g.set_entry_point("welcome")
g.add_edge("welcome", "setup")
g.add_conditional_edges("setup", route_after_setup)
g.add_edge("checkpoint", "architecture")
g.add_edge("architecture", END)

return g.compile(interrupt_before=["checkpoint"])
```

**Try it:**

```bash
python -m src.graph Alex backend
```

---

## MCP Server

### MCP1 (core 🟡) — Expose tools as an MCP server · ⏱️ ~8 min

**File:** `src/mcp_server.py` → `create_mcp_server()`
**Goal:** wrap your tools as an MCP server so any MCP-aware client (Cursor,
Claude Code) can call them.

**What you'll use:**

- `from mcp.server.fastmcp import FastMCP`
- `@mcp.tool()` decorator — same idea as LangChain's `@tool`

**Definition of done** (`python -m checks.check_mcp`, section MCP1):

- `create_mcp_server()` returns a `FastMCP` instance
- The server has `search_docs`, `read_file`, and `git_blame` tools registered

**Hints:**

```python
from mcp.server.fastmcp import FastMCP
from src.retrieve import semantic_search

mcp = FastMCP("onboarding-assistant")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search team docs and code for the given query."""
    hits = semantic_search(query, k=4)
    return "\n\n".join(f"{h['source']}:\n{h['text']}" for h in hits)
```

**Connect to Cursor:** add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "onboarding-assistant": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "<repo-root>"
    }
  }
}
```

### MCP2 (explore 🔵) — Add MCP resources · ⏱️ ~5 min

**File:** `src/mcp_server.py` → `add_resources()`
**Goal:** add read-only data endpoints (resources) to the MCP server.

**Hints:**

```python
@mcp.resource("onboarding://checklist")
def get_checklist() -> str:
    return "1. Clone repo\n2. Run setup\n3. ..."
```

Resources are different from tools: they're read-only data that the client can
fetch without the model making a decision. Think of them as context the IDE
can pre-load.

---

## LangSmith

### LS1 (core 🟢) — Configure and verify LangSmith · ⏱️ ~5 min

**File:** `src/langsmith_utils.py` → `ensure_langsmith_configured()` + `get_langsmith_client()`
**Goal:** set up LangSmith tracing so you can see inside everything you just
built. By this point you've already run the agent and graph — now you'll
configure LangSmith and see those traces in the dashboard.

**Pre-requisite:** sign up at [smith.langchain.com](https://smith.langchain.com/)
(free tier is fine) and get your API key.

**Step 0 — Set the env vars** in your `.env` (if not already done in setup):

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-pt-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=onboarding-assistant
```

**Definition of done** (`python -m checks.check_langsmith`, section LS1):

- `ensure_langsmith_configured()` returns a status dict with `tracing_enabled`,
  `api_key_set`, `project`, `endpoint`, and `status`
- `status` is `"ok"` when properly configured, `"disabled"` when tracing is off,
  `"misconfigured"` when the API key is missing
- `get_langsmith_client()` returns a `langsmith.Client` instance

**Hints:**

```python
import os
from src import config

def ensure_langsmith_configured() -> dict:
    tracing_enabled = config.LANGSMITH_TRACING_ENABLED
    api_key_set = bool(config.LANGSMITH_API_KEY) and config.LANGSMITH_API_KEY != "lsv2-..."
    if not tracing_enabled:
        status = "disabled"
    elif not api_key_set:
        status = "misconfigured"
    else:
        status = "ok"
    return {
        "tracing_enabled": tracing_enabled,
        "api_key_set": api_key_set,
        "project": config.LANGSMITH_PROJECT,
        "endpoint": config.LANGSMITH_ENDPOINT,
        "status": status,
    }
```

**Verify it:**

```bash
python -m checks.check_langsmith
python -m src.langsmith_utils        # prints your config status
```

**Now go explore:** open [smith.langchain.com](https://smith.langchain.com/) and
look at the traces from running the agent (A3) and graph (G3). You should see:

- The full agent trace (input question → tool calls → final answer)
- Each tool that was called and what it returned
- The graph trace with each node as a span
- Latency breakdown per step

### LS2 (core 🟡) — Custom tracing with @traceable · ⏱️ ~8 min

**File:** `src/langsmith_utils.py` → `traced_retrieval()` + `traced_format_context()`
**Goal:** LangChain agents and LangGraph auto-trace, but your own functions
(retrieval, context formatting, business logic) are invisible by default. The
`@traceable` decorator adds them to the trace tree.

**Why this matters:** without `@traceable`, you see the LLM call but not *what
context was assembled*, or *what the retriever returned*. That makes it
impossible to debug "the answer was wrong because the retrieval missed the
relevant chunk."

**What you'll use:**

- `from langsmith import traceable`
- `@traceable(run_type="retriever", name="semantic_search")` — for retrieval
- `@traceable(run_type="chain", name="format_context")` — for formatting

**Definition of done** (`python -m checks.check_langsmith`, section LS2):

- `traced_retrieval("setup", k=2)` returns results with `{text, source, score}`
- `traced_format_context(hits)` returns a formatted string with `[1]`, `[2]` citations

**Hints:**

```python
from langsmith import traceable
from src.retrieve import semantic_search

@traceable(run_type="retriever", name="semantic_search")
def traced_retrieval(query: str, k: int = 4) -> list[dict]:
    return semantic_search(query, k)

@traceable(run_type="chain", name="format_context")
def traced_format_context(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)
```

**run_type options:** `"retriever"` renders as a retrieval step in the dashboard
(with document viewer), `"chain"` is a generic processing step, `"llm"` is for
model calls, `"tool"` is for tool invocations. Pick the one that matches what
your function does.

---

## Further ideas (if time permits)

- **Agent memory** — give the agent `SessionMemory` so follow-up questions work
  across tool calls.
- **Multi-step graph** — extend the LangGraph flow with a "first ticket" node that
  assigns a starter task and a "first PR" node that reviews their code.
- **LangSmith evaluation** — use LangSmith's evaluation framework to build a
  golden-set test for agent quality (latency, tool usage, answer correctness).
- **Prompt versioning** — use `client.push_prompt()` / `client.pull_prompt()` to
  version the agent's system prompt and swap it without code changes.
