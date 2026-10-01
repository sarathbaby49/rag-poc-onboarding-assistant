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
**Goal:** create an agent graph that binds the three tools to a chat model.

**What you'll use:**

- `ChatOpenAI` from `langchain_openai` — talks to the LiteLLM proxy
- `create_agent` from `langchain.agents` — the modern LangChain agent builder

**Definition of done** (`python -m checks.check_agent`, section A2):

- `build_agent()` returns an object with `.invoke()`

**Hints:**

```python
from langchain.agents import create_agent

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    base_url=config.LITELLM_PROXY_API_BASE or None,
    api_key=config.LITELLM_PROXY_API_KEY or None,
)

return create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)
```

### A3 (core 🟢) — Run the agent · ⏱️ ~3 min

**File:** `src/agent.py` → `run_agent()`
**Goal:** invoke the agent on a question and return the answer string.

**Definition of done** (`python -m checks.check_agent`, section A3):

- `run_agent("What files are in this repo?")` returns a non-empty string

**Hint:**

```python
agent = build_agent()
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
return result["messages"][-1].content
```

**Try it:**

```bash
python -m src.agent "Where is the payment provider code and who last changed it?"
```

---

## LangGraph Flow — Onboarding Plan Chatbot

A new hire asks for a 2-week onboarding plan. The graph drafts it, grounds
every step in real docs (keyword-based RAG), loops back to re-plan when a
step has no supporting doc, then pauses for a mentor to approve or reject.

```
START → plan → retrieve → check_grounding ─┬─► mentor_approval ─┬─► publish → END
                                            │                    │
                                            └─► replan ──────────┘
                                               (gaps / rejected)
```

### G1 (core 🟡) — Implement node functions · ⏱️ ~15 min

**File:** `src/graph.py` — six nodes, each receives `PlanState`, returns a
partial update dict.

**Helpers already provided:** `_parse_request()`, `_make_plan()`,
`_keyword_search()`, `_load_docs()`, `_TEMPLATE_PLAN`, `_llm_plan()`,
`_llm_replan()`, `_llm_call()`, `_get_llm()`.

| Node | What it does | Key return fields |
|------|-------------|-------------------|
| **G1a `plan`** | Parse role/days, **LLM-generate** a role-specific plan, greet user | `role`, `days`, `plan`, `messages` |
| **G1b `retrieve`** | For each step, keyword-search docs and set `source` | `plan` (updated) |
| **G1c `check_grounding`** | List steps where `source is None` | `gaps` |
| **G1d `replan`** | **LLM-fix** un-grounded steps + apply mentor feedback; increment `attempts` | `plan`, `gaps` (empty), `mentor_feedback` (""), `attempts` |
| **G1e `mentor_approval`** | Call `interrupt()` → "approve" or feedback | `approved`, `mentor_feedback`, `messages` |
| **G1f `publish`** | Format the final plan as a nice message | `messages` |

**Definition of done** (`python -m checks.check_graph`, section G1):

- `plan` parses role, sets days, creates a non-empty plan, appends a welcome message
- `retrieve` finds sources for most steps; the "compliance training" step stays `None`
- `check_grounding` finds the compliance-training gap
- `replan` increments attempts and clears gaps
- `publish` produces a message with day information

**Hints — `plan`** (uses LLM via `_llm_plan`, falls back to `_make_plan`):

```python
def plan(state: PlanState) -> dict:
    role, days = _parse_request(state["user_request"])
    doc_names = list(_load_docs().keys())
    draft = _llm_plan(role, days, doc_names)   # LLM-generated, role-specific
    return {
        "role": role, "days": days, "plan": draft,
        "messages": state["messages"] + [f"👋 Welcome! Building a {days}-day plan for a {role} dev…"],
    }
```

**Hints — `replan`** (uses LLM via `_llm_replan`, falls back to rule-based):

```python
def replan(state: PlanState) -> dict:
    doc_names = list(_load_docs().keys())
    llm_result = _llm_replan(state["plan"], state["gaps"], state.get("mentor_feedback", ""), doc_names)
    new_plan = llm_result if llm_result is not None else ...  # rule-based fallback
    return {"plan": new_plan, "gaps": [], "mentor_feedback": "", "attempts": state["attempts"] + 1}
```

**Hints — `mentor_approval`:**

```python
from langgraph.types import interrupt
response = interrupt({"question": "Review this plan", "plan": plan_text})
if response.strip().lower() == "approve":
    return {"approved": True, "messages": state["messages"] + ["✅ Approved!"]}
return {"approved": False, "mentor_feedback": response, "messages": state["messages"] + [f"📝 Feedback: {response}"]}
```

### G2 (core 🟢) — Routing functions · ⏱️ ~3 min

**File:** `src/graph.py` → `route_after_check()`, `route_after_mentor()`

| Router | Logic |
|--------|-------|
| `route_after_check` | gaps AND attempts < MAX_ATTEMPTS → `"replan"`, else → `"mentor_approval"` |
| `route_after_mentor` | approved → `"publish"`, else → `"replan"` |

**Hint:** each is 1-2 lines.

### G3 (core 🟡) — Assemble the graph · ⏱️ ~8 min

**File:** `src/graph.py` → `build_graph(checkpointer=None)`

**What you'll use:**

- `from langgraph.graph import StateGraph, START, END`
- `g.add_node(name, function)`
- `g.add_edge(from, to)`
- `g.add_conditional_edges(from, routing_fn, mapping)`
- `g.compile(checkpointer=...)`

**Definition of done** (`python -m checks.check_graph`, section G3):

- `build_graph()` returns a compiled graph with all 6 nodes.

**Hints:**

```python
g = StateGraph(PlanState)
g.add_node("plan", plan)
g.add_node("retrieve", retrieve)
g.add_node("check_grounding", check_grounding)
g.add_node("replan", replan)
g.add_node("mentor_approval", mentor_approval)
g.add_node("publish", publish)

g.add_edge(START, "plan")
g.add_edge("plan", "retrieve")
g.add_edge("retrieve", "check_grounding")
g.add_conditional_edges("check_grounding", route_after_check, {
    "replan": "replan",
    "mentor_approval": "mentor_approval",
})
g.add_edge("replan", "retrieve")
g.add_conditional_edges("mentor_approval", route_after_mentor, {
    "publish": "publish",
    "replan": "replan",
})
g.add_edge("publish", END)

if checkpointer is None:
    checkpointer = MemorySaver()
return g.compile(checkpointer=checkpointer)
```

**Try it in the CLI:**

```bash
python -m src.graph
```

**Or in the orchestration lab** (graph mode) — type a request like
"I'm a frontend dev, plan my first 2 weeks". When the mentor prompt appears,
type "approve" or give feedback like "add a pairing session on Day 3".

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
