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

import importlib
import sys
import traceback

import streamlit as st
from dotenv import load_dotenv

from src import config

load_dotenv()


def _reload(module_name: str):
    """Force-reload an src module so Streamlit picks up edits during exercises."""
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


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
    "Graph · onboarding plan": ("graph", "plan → ground → replan → mentor approve → publish"),
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
        st.caption(
            "The graph plans an onboarding schedule, grounds each step in "
            "real docs, loops to fix gaps, then pauses for mentor approval."
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
    st.session_state.pop("graph_checkpointer", None)
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
    mod = _reload("src.agent")
    answer = mod.run_agent(prompt)
    return {"answer": answer, "sources": [], "extra": None}


# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH mode
# ═════════════════════════════════════════════════════════════════════════════
def _run_graph(prompt: str) -> dict:
    """Run one graph turn and return result dict with 'answer', 'extra', 'waiting'."""
    mod = _reload("src.graph")

    # Keep one MemorySaver per session so the checkpoint survives across turns
    if "graph_checkpointer" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph_checkpointer = MemorySaver()

    prev_state = st.session_state.graph_state

    state, new_msgs, waiting = mod.run_graph_turn(
        user_message=prompt,
        graph_state=prev_state,
        thread_id="streamlit",
        checkpointer=st.session_state.graph_checkpointer,
    )

    st.session_state.graph_state = state

    answer = "\n\n---\n\n".join(new_msgs) if new_msgs else "(no response from the graph)"

    # Status line
    attempts = state.get("attempts", 0)
    gaps = state.get("gaps", [])
    approved = state.get("approved", False)
    if waiting:
        extra = f"⏸️ **Waiting for mentor approval** · Attempts: {attempts}"
    elif approved:
        extra = f"✅ **Plan published** · Attempts: {attempts}"
    else:
        extra = f"Attempts: {attempts} · Gaps: {len(gaps)}"

    return {"answer": answer, "sources": [], "extra": extra, "waiting": waiting}


def _handle_graph(prompt: str) -> dict:
    return _run_graph(prompt)


# ═════════════════════════════════════════════════════════════════════════════
#  MCP mode
# ═════════════════════════════════════════════════════════════════════════════
def _handle_mcp(prompt: str) -> dict:
    mcp_mod = _reload("src.mcp_server")
    server = mcp_mod.create_mcp_server()

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
    ls_mod = _reload("src.langsmith_utils")

    # "config" / "status" → run ensure_langsmith_configured
    if any(kw in lower for kw in ("config", "status", "check", "verify", "setup")):
        status = ls_mod.ensure_langsmith_configured()
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
        client = ls_mod.get_langsmith_client()
        return {"answer": f"✅ LangSmith client created: `{type(client).__name__}`", "sources": [], "extra": None}

    # "format" → test traced_format_context with sample data
    if "format" in lower:
        sample = [
            {"text": "Setup: clone the repo and run pip install.", "source": "setup.md", "score": 0.95},
            {"text": "Auth uses JWT with refresh tokens.", "source": "auth.md", "score": 0.82},
        ]
        formatted = ls_mod.traced_format_context(sample)
        return {"answer": f"**`traced_format_context()`**\n\n```\n{formatted}\n```", "sources": [], "extra": None}

    # Default: treat as a traced_retrieval query
    hits = ls_mod.traced_retrieval(prompt, k=4)
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
    "graph": "e.g. I'm a mid-level frontend dev, plan my first 2 weeks  |  approve  |  add pairing on Day 3",
    "mcp": "e.g. payment providers  |  read src/config.py  |  blame src/config.py 1",
    "langsmith": "e.g. config  |  client  |  format  |  payment providers (traced search)",
}

# ── Chat UI ──────────────────────────────────────────────────────────────────

# Graph mode — show intro prompt if no history yet
if mode == "graph" and not st.session_state.history and st.session_state.graph_state is None:
    st.session_state.history.append({
        "role": "assistant",
        "content": (
            "👋 **Onboarding Plan Chatbot** — tell me your role and how many "
            "days/weeks to plan, and I'll build a grounded onboarding schedule.\n\n"
            "_Example: \"I'm a mid-level frontend dev, plan my first 2 weeks\"_"
        ),
    })

# Replay conversation
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("extra"):
            st.caption(turn["extra"])

# ---------- Graph: mentor approval buttons ----------
_graph_waiting = (
    mode == "graph"
    and st.session_state.graph_state is not None
    and st.session_state.graph_state.get("_waiting_for_mentor", False)
)

if _graph_waiting:
    st.divider()
    st.markdown("**🧑‍🏫 Mentor Review (Sarath)**")
    col_approve, col_reject = st.columns([1, 3])
    with col_approve:
        approve_clicked = st.button("✅ Approve", type="primary", use_container_width=True)
    with col_reject:
        feedback_text = st.text_input(
            "Or give feedback:",
            placeholder="e.g. add a pairing session on Day 3",
            key="mentor_feedback_input",
            label_visibility="collapsed",
        )
        reject_clicked = st.button("📝 Send feedback", use_container_width=True, disabled=not feedback_text)

    if approve_clicked:
        st.session_state.history.append({"role": "user", "content": "✅ Mentor: approve"})
        with st.chat_message("assistant"):
            with st.spinner("Mentor approved — publishing…"):
                result = _safe_call(_run_graph, "approve")
            st.markdown(result["answer"])
            if result.get("extra"):
                st.caption(result["extra"])
        entry = {"role": "assistant", "content": result["answer"]}
        if result.get("extra"):
            entry["extra"] = result["extra"]
        st.session_state.history.append(entry)
        st.rerun()

    if reject_clicked and feedback_text:
        st.session_state.history.append({"role": "user", "content": f"📝 Mentor: {feedback_text}"})
        with st.chat_message("assistant"):
            with st.spinner("Applying feedback — re-planning…"):
                result = _safe_call(_run_graph, feedback_text)
            st.markdown(result["answer"])
            if result.get("extra"):
                st.caption(result["extra"])
        entry = {"role": "assistant", "content": result["answer"]}
        if result.get("extra"):
            entry["extra"] = result["extra"]
        st.session_state.history.append(entry)
        st.rerun()

# ---------- Normal chat input (hidden while waiting for mentor in graph mode) ----------
if not _graph_waiting:
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

        # If this was a graph turn that's now waiting, rerun to show buttons
        if mode == "graph" and result.get("waiting"):
            st.rerun()
