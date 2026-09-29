"""Orchestration Lab — a Streamlit chat UI to test the Layer 4 exercises.

    streamlit run orchestration_lab.py

Pick which POC to test in the sidebar, then chat. Unfinished exercises show a
friendly message instead of crashing — just like rag_app.py does for G1–G4.

Modes:
  - **Agent**     — ask the setup-troubleshooting agent a question (A1–A3)
  - **Graph**     — walk through the LangGraph onboarding flow (G1–G3)
  - **MCP**       — call the MCP server's tools directly (MCP1–MCP2)
  - **LangSmith** — test @traceable wrappers + check config (LS1–LS2)
"""

from __future__ import annotations

import traceback

import streamlit as st
from dotenv import load_dotenv

from src import config

load_dotenv()

st.set_page_config(page_title="Orchestration Lab", page_icon="🤖")

st.markdown(
    """
    <style>
      div[role="radiogroup"] > label { align-items: flex-start; }
      div[role="radiogroup"] > label > div:first-child { margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Orchestration Lab")
st.caption("Test your Layer 4 exercises — agents, graphs, MCP, and LangSmith.")

# ── Modes ────────────────────────────────────────────────────────────────────
MODES = {
    "Agent · run_agent": ("agent", "ask the setup-troubleshooting agent a question"),
    "Graph · onboarding flow": ("graph", "walk through the LangGraph onboarding flow"),
    "MCP · tool calls": ("mcp", "call MCP server tools (search_docs, read_file, git_blame)"),
    "LangSmith · tracing": ("langsmith", "test @traceable wrappers + check config"),
}

with st.sidebar:
    st.header("Which POC?")
    labels = list(MODES.keys())
    mode_label = st.radio(
        "Function under test",
        labels,
        captions=[MODES[label][1] for label in labels],
    )
    mode = MODES[mode_label][0]

    st.divider()
    if st.button("🧹 Clear conversation"):
        st.session_state.history = []
        st.session_state.graph_state = None
        st.rerun()

    if mode == "graph":
        st.divider()
        st.text_input("Joinee name", value="Alex", key="joinee_name")
        st.selectbox("Role", ["backend", "frontend", "fullstack", "data"], key="joinee_role")
        st.caption(
            "The graph walks through: welcome → setup → architecture. "
            "Type a message with 'error' or 'stuck' to trigger the mentor path."
        )


# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

# Clear history when switching modes
if st.session_state.last_mode != mode:
    st.session_state.history = []
    st.session_state.graph_state = None
    st.session_state.last_mode = mode


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_call(fn, *args, **kwargs) -> dict:
    """Call fn and return {"answer": str, "sources": list, "extra": str | None}.

    Catches NotImplementedError and exceptions gracefully.
    """
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, str):
            return {"answer": result, "sources": [], "extra": None}
        return result
    except NotImplementedError as e:
        return {
            "answer": f"⚠️ **Not implemented yet** — {e}\n\nFinish the exercise, then try again.",
            "sources": [],
            "extra": None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "answer": f"❌ **Error:** {type(e).__name__}: {e}",
            "sources": [],
            "extra": tb,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT mode
# ═════════════════════════════════════════════════════════════════════════════
def _handle_agent(prompt: str) -> dict:
    from src.agent import run_agent
    answer = run_agent(prompt)
    return {"answer": answer, "sources": [], "extra": None}


# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH mode
# ═════════════════════════════════════════════════════════════════════════════
def _handle_graph(prompt: str) -> dict:
    from src.graph import build_graph, OnboardingState

    graph = build_graph()

    if st.session_state.graph_state is None:
        # First message — start the flow from welcome
        initial_state: OnboardingState = {
            "joinee": st.session_state.get("joinee_name", "Alex"),
            "role": st.session_state.get("joinee_role", "backend"),
            "stage": "welcome",
            "messages": [{"role": "user", "content": prompt}] if prompt else [],
            "needs_mentor": False,
            "mentor_notes": "Reviewed and approved.",
            "completed_steps": [],
        }
        result = graph.invoke(initial_state)
    else:
        # Continue — feed the new user message into setup
        state = dict(st.session_state.graph_state)
        state["messages"] = state["messages"] + [{"role": "user", "content": prompt}]
        # Re-enter at setup stage so the graph processes the new message
        if state.get("stage") == "setup":
            result = graph.invoke(state)
        else:
            # Flow already completed — just echo
            return {
                "answer": "🎉 The onboarding flow is complete! Click **Clear conversation** to restart.",
                "sources": [],
                "extra": None,
            }

    st.session_state.graph_state = result

    # Collect the new assistant messages
    new_msgs = [m for m in result.get("messages", []) if m.get("role") == "assistant"]
    answer_parts = [m["content"] for m in new_msgs]
    answer = "\n\n---\n\n".join(answer_parts) if answer_parts else "(no response from the graph)"

    stage = result.get("stage", "?")
    completed = result.get("completed_steps", [])
    extra = f"**Stage:** {stage} · **Completed:** {' → '.join(completed) if completed else '—'}"
    if result.get("needs_mentor"):
        extra += " · ⚠️ Needs mentor review"

    return {"answer": answer, "sources": [], "extra": extra}


# ═════════════════════════════════════════════════════════════════════════════
#  MCP mode
# ═════════════════════════════════════════════════════════════════════════════
def _handle_mcp(prompt: str) -> dict:
    from src.mcp_server import create_mcp_server
    import src.mcp_server as mcp_mod

    # Create the server to verify it works
    server = create_mcp_server()

    # Parse the prompt to figure out which tool to call
    lower = prompt.lower().strip()

    if lower.startswith("read ") or lower.startswith("read_file "):
        path = prompt.split(maxsplit=1)[1].strip() if " " in prompt else ""
        if hasattr(mcp_mod, "read_file"):
            result = mcp_mod.read_file(path)
        else:
            from pathlib import Path
            resolved = (config.BASE_DIR / path).resolve()
            result = resolved.read_text() if resolved.exists() else f"File not found: {path}"
        return {"answer": f"**`read_file(\"{path}\")`**\n\n```\n{result[:3000]}\n```", "sources": [], "extra": None}

    if lower.startswith("blame ") or lower.startswith("git_blame "):
        parts = prompt.split()
        path = parts[1] if len(parts) > 1 else ""
        line = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
        if hasattr(mcp_mod, "git_blame"):
            result = mcp_mod.git_blame(path, line)
        else:
            result = "(git_blame not found on mcp_server module)"
        return {"answer": f"**`git_blame(\"{path}\", {line})`**\n\n```\n{result}\n```", "sources": [], "extra": None}

    # Default: treat as search_docs query
    if hasattr(mcp_mod, "search_docs"):
        result = mcp_mod.search_docs(prompt)
    else:
        from src.retrieve import semantic_search
        hits = semantic_search(prompt, k=4)
        result = "\n\n".join(f"📄 {h['source']} (score: {h['score']:.2f}):\n{h['text']}" for h in hits) if hits else "No results."

    return {"answer": f"**`search_docs(\"{prompt}\")`**\n\n{result}", "sources": [], "extra": None}


# ═════════════════════════════════════════════════════════════════════════════
#  LANGSMITH mode
# ═════════════════════════════════════════════════════════════════════════════
def _handle_langsmith(prompt: str) -> dict:
    lower = prompt.lower().strip()

    # "config" / "status" → run ensure_langsmith_configured
    if any(kw in lower for kw in ("config", "status", "check", "verify", "setup")):
        from src.langsmith_utils import ensure_langsmith_configured
        status = ensure_langsmith_configured()
        lines = [
            "**LangSmith Configuration:**",
            f"- Tracing enabled: {'✅' if status['tracing_enabled'] else '❌'} `{status['tracing_enabled']}`",
            f"- API key set: {'✅' if status['api_key_set'] else '❌'} `{status['api_key_set']}`",
            f"- Project: `{status['project']}`",
            f"- Endpoint: `{status['endpoint']}`",
            f"- Status: **{status['status']}**",
        ]
        return {"answer": "\n".join(lines), "sources": [], "extra": None}

    # "client" → test get_langsmith_client
    if "client" in lower:
        from src.langsmith_utils import get_langsmith_client
        client = get_langsmith_client()
        return {"answer": f"✅ LangSmith client created: `{type(client).__name__}`", "sources": [], "extra": None}

    # "format" → test traced_format_context with sample data
    if "format" in lower:
        from src.langsmith_utils import traced_format_context
        sample = [
            {"text": "Setup: clone the repo and run pip install.", "source": "setup.md", "score": 0.95},
            {"text": "Auth uses JWT with refresh tokens.", "source": "auth.md", "score": 0.82},
        ]
        formatted = traced_format_context(sample)
        return {"answer": f"**`traced_format_context()`**\n\n```\n{formatted}\n```", "sources": [], "extra": None}

    # Default: treat as a traced_retrieval query
    from src.langsmith_utils import traced_retrieval
    hits = traced_retrieval(prompt, k=4)
    if not hits:
        return {"answer": "No results — make sure the vector index is built (`python -m src.ingest`).", "sources": [], "extra": None}

    lines = [f"**`traced_retrieval(\"{prompt}\")`**\n"]
    for i, h in enumerate(hits, 1):
        lines.append(f"**[{i}] {h['source']}** — score {h['score']:.2f}")
        lines.append(f"```\n{h['text'][:300]}\n```")
    return {"answer": "\n".join(lines), "sources": hits, "extra": None}


# ── Dispatch ─────────────────────────────────────────────────────────────────
HANDLERS = {
    "agent": _handle_agent,
    "graph": _handle_graph,
    "mcp": _handle_mcp,
    "langsmith": _handle_langsmith,
}

PLACEHOLDERS = {
    "agent": "e.g. Where is the payment provider code and who last changed it?",
    "graph": "e.g. How do I set up my local environment?  (try 'I got an error' to trigger mentor)",
    "mcp": "e.g. payment providers  |  read src/config.py  |  blame src/config.py 1",
    "langsmith": "e.g. config  |  client  |  format  |  payment providers (traced search)",
}

# ── Chat UI ──────────────────────────────────────────────────────────────────

# Auto-start the graph flow on first visit
if mode == "graph" and not st.session_state.history and st.session_state.graph_state is None:
    result = _safe_call(_handle_graph, "")
    if "Not implemented" not in result["answer"] and "Error" not in result["answer"]:
        st.session_state.history.append({"role": "assistant", "content": result["answer"]})
        if result.get("extra"):
            st.session_state.history[-1]["extra"] = result["extra"]
    else:
        # Show the error as the first message
        st.session_state.history.append({"role": "assistant", "content": result["answer"]})

# Replay conversation
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("extra"):
            st.caption(turn["extra"])

if prompt := st.chat_input(PLACEHOLDERS.get(mode, "Ask something…")):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working…"):
            result = _safe_call(HANDLERS[mode], prompt)

        st.markdown(result["answer"])
        if result.get("extra"):
            st.caption(result["extra"])

    entry = {"role": "assistant", "content": result["answer"]}
    if result.get("extra"):
        entry["extra"] = result["extra"]
    st.session_state.history.append(entry)
