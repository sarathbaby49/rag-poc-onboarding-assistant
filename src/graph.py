"""Layer 4 — LangGraph "Onboarding Plan Chatbot" (YOUR EXERCISES: G1 + G2 + G3).

A new hire asks for a 2-week onboarding plan. The graph:
  1. **plans** — drafts a day-by-day plan
  2. **retrieves** — grounds each step in real docs (keyword search)
  3. **checks** — finds gaps (steps with no supporting doc)
  4. **replans** — fixes gaps or applies mentor feedback  (loop)
  5. **mentor approval** — pauses for a human mentor to approve / reject
  6. **publishes** — formats the final plan

    START → plan → retrieve → check_grounding ─┬─► mentor_approval ─┬─► publish → END
                                                │                    │
                                                └─► replan ──────────┘
                                                   (gaps / rejected)

Concepts covered:
  - StateGraph + TypedDict shared state
  - Conditional edges (route_after_check, route_after_mentor)
  - Loops (replan → retrieve → check_grounding → …)
  - Human-in-the-loop with interrupt() + Command(resume=…)
  - MemorySaver checkpointer for pause / resume

Self-check:  python -m checks.check_graph
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src import config

# ---------------------------------------------------------------------------
# Constants (provided)
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3  # Safety valve — prevents infinite replan loops.

DOCS_DIR = Path(config.BASE_DIR) / "data" / "sample_company"


# ---------------------------------------------------------------------------
# State schema (provided)
# ---------------------------------------------------------------------------

class PlanState(TypedDict):
    user_request: str          # the new hire's free-text request
    role: str                  # e.g. "frontend", "backend"
    days: int                  # how many working days to plan (default 10)
    plan: list[dict]           # [{"day": 1, "task": "...", "source": "..." | None}]
    gaps: list[str]            # tasks with no supporting doc
    mentor_feedback: str       # free-text from the mentor (empty = none)
    approved: bool             # True once the mentor approves
    attempts: int              # how many times we've replanned
    messages: list[str]        # bot messages shown to the user


# ---------------------------------------------------------------------------
# Helpers (provided)
# ---------------------------------------------------------------------------

def _load_docs() -> dict[str, str]:
    """Load every .md file under DOCS_DIR into {filename: content}."""
    docs: dict[str, str] = {}
    if not DOCS_DIR.exists():
        return docs
    for p in sorted(DOCS_DIR.rglob("*.md")):
        docs[p.name] = p.read_text(errors="replace")
    return docs


def _keyword_search(task: str, docs: dict[str, str]) -> str | None:
    """Return the first doc filename whose content matches keywords from *task*.

    Simple keyword overlap — no vector DB needed.
    """
    words = set(re.findall(r"[a-z]{3,}", task.lower()))
    words -= {"the", "and", "for", "with", "your", "from", "about", "read",
              "learn", "review", "run", "set", "day", "get", "how", "use"}
    best_name, best_score = None, 0
    for name, content in docs.items():
        lower = content.lower()
        score = sum(1 for w in words if w in lower)
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= 1 else None


# ---------------------------------------------------------------------------
# LLM helpers (provided)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        max_tokens=config.MAX_TOKENS,
    )


def _llm_call(system: str, user: str) -> str:
    """Send a system + user prompt to the LLM and return the text response."""
    llm = _get_llm()
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return resp.content


def _parse_request(text: str) -> tuple[str, int]:
    """Extract (role, days) from user's free-text request using the LLM.

    Falls back to regex if the LLM is unavailable.
    """
    # Try LLM-based extraction first
    try:
        raw = _llm_call(
            system=(
                "Extract the developer role and number of working days from the user's request.\n"
                "Rules:\n"
                '- Return ONLY JSON: {"role": "...", "days": N}\n'
                "- role must be one of: frontend, backend, fullstack, data, devops, mobile\n"
                "- Convert weeks to working days (1 week = 5 days, 2 weeks = 10 days)\n"
                '- "first week" or "a week" = 5 days\n'
                "- If no duration mentioned, default to 10 days\n"
                "- If no role mentioned, default to backend\n"
                "- No markdown, no explanation — just the JSON"
            ),
            user=text,
        )
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        parsed = json.loads(cleaned)
        role = parsed.get("role", "backend").lower().replace("full-stack", "fullstack")
        days = int(parsed.get("days", 10))
        if role and 1 <= days <= 30:
            return role, days
    except Exception:
        pass

    # Fallback: regex-based extraction
    role = "backend"
    for r in ("frontend", "fullstack", "full-stack", "data", "backend", "devops", "mobile"):
        if r in text.lower():
            role = r.replace("full-stack", "fullstack")
            break
    days_match = re.search(r"(\d+)\s*(?:day|week)", text.lower())
    if days_match:
        n = int(days_match.group(1))
        days = n * 5 if "week" in text.lower()[days_match.start():] else n
    else:
        days = 10
    return role, days


# ---------------------------------------------------------------------------
# Default (deterministic) plan template
# ---------------------------------------------------------------------------

_TEMPLATE_PLAN = [
    "Read the repo README and architecture overview",
    "Set up local environment (clone, install, run)",
    "Read authentication docs and trace login flow",
    "Complete mandatory compliance training",            # ← deliberately un-groundable
    "Read the CI/CD pipeline overview",
    "Walk through the payments module",
    "Read frontend conventions and component structure",
    "Pick your first ticket and create a branch",
    "Implement and test your first change",
    "Open your first PR and request review",
]


def _make_plan(role: str, days: int) -> list[dict]:
    """Build a day-by-day plan (deterministic fallback, trim to *days*)."""
    tasks = _TEMPLATE_PLAN[:days]
    return [{"day": i + 1, "task": t, "source": None} for i, t in enumerate(tasks)]


def _llm_plan(role: str, days: int, doc_names: list[str]) -> list[dict]:
    """Use the LLM to generate a role-specific onboarding plan.

    Falls back to _make_plan() if LLM response can't be parsed.
    """
    system = (
        "You are an onboarding planner for Acme Shop, a FastAPI + PostgreSQL + Redis "
        "e-commerce backend. Generate a day-by-day onboarding plan.\n\n"
        f"Available docs for reference: {', '.join(doc_names)}\n\n"
        "Rules:\n"
        "- Return ONLY a JSON array of objects: [{\"day\": 1, \"task\": \"...\"}]\n"
        "- One task per day, concise (under 15 words each)\n"
        "- Include at least one task that has NO matching doc (e.g. compliance training, "
        "shadow on-call) so the grounding loop is exercised\n"
        "- Tailor tasks to the role (e.g. frontend devs read frontend-conventions)\n"
        "- No markdown, no explanation — just the JSON array"
    )
    user = f"Role: {role}, Days: {days}"
    try:
        raw = _llm_call(system, user)
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        tasks = json.loads(cleaned)
        if isinstance(tasks, list) and all("day" in t and "task" in t for t in tasks):
            return [{"day": t["day"], "task": t["task"], "source": None} for t in tasks[:days]]
    except Exception:
        pass
    return _make_plan(role, days)


def _llm_replan(
    plan: list[dict],
    gaps: list[str],
    mentor_feedback: str,
    doc_names: list[str],
) -> list[dict] | None:
    """Use the LLM to fix un-grounded steps and apply mentor feedback.

    Returns None if parsing fails (caller should fall back to rule-based).
    """
    plan_text = "\n".join(f"Day {s['day']}: {s['task']} (source: {s['source'] or 'NONE'})" for s in plan)
    system = (
        "You are an onboarding planner for Acme Shop. Fix this plan.\n\n"
        f"Available docs: {', '.join(doc_names)}\n\n"
        "Rules:\n"
        "- Replace tasks marked 'source: NONE' with alternatives that relate to the available docs\n"
        "- If mentor feedback is given, apply it to the plan\n"
        "- Keep tasks that already have sources unchanged\n"
        "- Return ONLY a JSON array: [{\"day\": 1, \"task\": \"...\"}]\n"
        "- No markdown, no explanation — just the JSON array"
    )
    user_parts = [f"Current plan:\n{plan_text}"]
    if gaps:
        user_parts.append(f"Gaps (no supporting doc): {gaps}")
    if mentor_feedback:
        user_parts.append(f"Mentor feedback: {mentor_feedback}")

    try:
        raw = _llm_call(system, "\n".join(user_parts))
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        tasks = json.loads(cleaned)
        if isinstance(tasks, list) and all("day" in t and "task" in t for t in tasks):
            return [{"day": t["day"], "task": t["task"], "source": None} for t in tasks]
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# G1: Implement node functions
#
# Each node is a function(state: PlanState) -> dict (partial state update).
# LangGraph merges the returned dict into the running state.
#
# Self-check:  python -m checks.check_graph   (section G1)
# ═══════════════════════════════════════════════════════════════════════════

def plan(state: PlanState) -> dict:
    """EXERCISE G1a — Parse request, use LLM to draft a plan, greet the user.

    Use _parse_request() to get (role, days), then _llm_plan() to generate
    a role-specific plan via the LLM. It falls back to _make_plan() if the
    LLM fails.

    Return dict with: role, days, plan, messages (appended).
    """
    # TODO(G1a): implement plan node
    raise NotImplementedError("Exercise G1a: implement plan node — see EXERCISES.md")


def retrieve(state: PlanState) -> dict:
    """EXERCISE G1b — Ground each plan step in a real doc.

    For each step in state["plan"], call _keyword_search(step["task"], docs)
    and set the step's "source" to the result (filename or None).

    Return dict with: plan (updated with sources).
    """
    # TODO(G1b): implement retrieve node
    raise NotImplementedError("Exercise G1b: implement retrieve node — see EXERCISES.md")


def check_grounding(state: PlanState) -> dict:
    """EXERCISE G1c — Identify un-grounded steps.

    gaps = list of "Day N: task" strings where source is None.

    Return dict with: gaps.
    """
    # TODO(G1c): implement check_grounding node
    raise NotImplementedError("Exercise G1c: implement check_grounding node — see EXERCISES.md")


def replan(state: PlanState) -> dict:
    """EXERCISE G1d — Use LLM to fix gaps and apply mentor feedback.

    Use _llm_replan() to ask the LLM to replace un-grounded steps and
    apply mentor feedback. If _llm_replan() returns None (LLM failed),
    fall back to rule-based replacement.

    Rule-based fallback:
      - "Complete mandatory compliance training" → "Read the CI/CD pipeline overview"
      - If mentor_feedback mentions a day, update that day's task.

    Clear gaps and mentor_feedback; increment attempts.

    Return dict with: plan, gaps (empty), mentor_feedback (""), attempts.
    """
    # TODO(G1d): implement replan node
    raise NotImplementedError("Exercise G1d: implement replan node — see EXERCISES.md")


def mentor_approval(state: PlanState) -> dict:
    """EXERCISE G1e — Pause for mentor approval using interrupt().

    Call: from langgraph.types import interrupt
    response = interrupt({"question": "...", "plan": plan_text})

    If response is "approve" → approved=True + success message.
    Otherwise → approved=False, mentor_feedback=response + re-plan message.

    Return dict with: approved, (optionally mentor_feedback), messages.
    """
    # TODO(G1e): implement mentor_approval node
    raise NotImplementedError("Exercise G1e: implement mentor_approval node — see EXERCISES.md")


def publish(state: PlanState) -> dict:
    """EXERCISE G1f — Format and publish the final plan.

    Build a nice multi-line string with one line per day:
      Day 1: task  (source)
    Append it + a congrats message to state["messages"].

    Return dict with: messages (appended).
    """
    # TODO(G1f): implement publish node
    raise NotImplementedError("Exercise G1f: implement publish node — see EXERCISES.md")


# ═══════════════════════════════════════════════════════════════════════════
# G2: Routing functions
#
# These are called by add_conditional_edges to decide the next node.
#
# Self-check:  python -m checks.check_graph   (section G2)
# ═══════════════════════════════════════════════════════════════════════════

def route_after_check(state: PlanState) -> str:
    """EXERCISE G2a — Route after check_grounding.

    - If gaps exist AND attempts < MAX_ATTEMPTS → "replan"
    - Otherwise → "mentor_approval"
    """
    # TODO(G2a): implement route_after_check
    raise NotImplementedError("Exercise G2a: implement route_after_check — see EXERCISES.md")


def route_after_mentor(state: PlanState) -> str:
    """EXERCISE G2b — Route after mentor_approval.

    - If approved → "publish"
    - Otherwise → "replan"
    """
    # TODO(G2b): implement route_after_mentor
    raise NotImplementedError("Exercise G2b: implement route_after_mentor — see EXERCISES.md")


# ═══════════════════════════════════════════════════════════════════════════
# G3: Assemble the graph
#
# Self-check:  python -m checks.check_graph   (section G3)
# ═══════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None):
    """EXERCISE G3 — Wire the nodes and edges, compile with a checkpointer.

    Nodes: plan, retrieve, check_grounding, replan, mentor_approval, publish

    Edges:
      START → plan → retrieve → check_grounding
      check_grounding → (conditional: route_after_check)
        "replan"           → replan
        "mentor_approval"  → mentor_approval
      replan → retrieve   (loop back)
      mentor_approval → (conditional: route_after_mentor)
        "publish" → publish
        "replan"  → replan
      publish → END

    Compile with: checkpointer (MemorySaver if None provided).
    """
    # TODO(G3): build and return the compiled graph
    raise NotImplementedError("Exercise G3: build the graph — see EXERCISES.md")


# ---------------------------------------------------------------------------
# Helpers for the Streamlit UI (provided)
# ---------------------------------------------------------------------------

def run_graph_turn(
    user_message: str,
    graph_state: dict | None = None,
    thread_id: str = "default",
    checkpointer=None,
) -> tuple[dict, list[str], bool]:
    """Run one turn of the plan chatbot.

    Returns (state_dict, new_bot_messages, is_waiting_for_mentor).
    """
    from langgraph.types import Command

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    if graph_state is None:
        # First turn — kick off the graph
        initial = PlanState(
            user_request=user_message,
            role="",
            days=10,
            plan=[],
            gaps=[],
            mentor_feedback="",
            approved=False,
            attempts=0,
            messages=[],
        )
        prev_msgs: list[str] = []
        events = list(graph.stream(initial, config, stream_mode="updates"))
    elif graph_state.get("_waiting_for_mentor"):
        # Resume from mentor interrupt
        prev_msgs = graph_state.get("_prev_messages", [])
        events = list(graph.stream(
            Command(resume=user_message), config, stream_mode="updates",
        ))
    else:
        return graph_state, ["The plan has already been published! Clear the conversation to start over."], False

    snapshot = graph.get_state(config)
    final_state = dict(snapshot.values) if snapshot.values else {}

    all_msgs = final_state.get("messages", [])
    new_msgs = all_msgs[len(prev_msgs):]

    waiting = bool(snapshot.next)

    if waiting and not any("sent it to your mentor" in m for m in new_msgs):
        plan_preview = "\n".join(
            f"- **Day {s['day']}:** {s['task']}  _({s['source'] or '?'})_"
            for s in final_state.get("plan", [])
        )
        new_msgs.append(
            f"📋 Your plan is ready. I've sent it to your mentor, **Sarath**, for approval.\n\n{plan_preview}"
        )

    result = {
        **final_state,
        "_waiting_for_mentor": waiting,
        "_prev_messages": all_msgs,
    }
    return result, new_msgs, waiting


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Interactive terminal chatbot."""
    from langgraph.types import Command

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "cli"}}

    request = input("You: ").strip()
    if not request:
        request = "Hi, I'm joining Monday as a mid-level frontend dev. Can you plan my first 2 weeks?"

    initial = PlanState(
        user_request=request, role="", days=10, plan=[], gaps=[],
        mentor_feedback="", approved=False, attempts=0, messages=[],
    )

    prev_messages: list[str] = []

    for event in graph.stream(initial, config, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "check_grounding":
                gaps = updates.get("gaps", [])
                dest = "replan" if gaps else "mentor_approval"
                print(f"\033[2m[{node_name}] gaps={gaps} -> {dest}\033[0m")
            elif node_name == "replan":
                print(f"\033[2m[{node_name}] attempts={updates.get('attempts', '?')}\033[0m")
            elif node_name == "retrieve":
                no_source = sum(1 for s in updates.get("plan", []) if s.get("source") is None)
                print(f"\033[2m[{node_name}] {no_source} step(s) have no source\033[0m")
            else:
                print(f"\033[2m[{node_name}]\033[0m")
            for msg in updates.get("messages", [])[len(prev_messages):] if "messages" in updates else []:
                print(f"BOT: {msg}")
            if "messages" in updates:
                prev_messages = updates["messages"]

    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        state = snapshot.values
        print("\n--- Plan for mentor review ---")
        for s in state.get("plan", []):
            print(f"  Day {s['day']}: {s['task']}  ({s['source'] or '?'})")
        mentor_input = input("\nMENTOR (Sarath) — type 'approve' or give feedback: ").strip()

        for event in graph.stream(Command(resume=mentor_input), config, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name == "check_grounding":
                    gaps = updates.get("gaps", [])
                    print(f"\033[2m[{node_name}] gaps={gaps}\033[0m")
                elif node_name == "replan":
                    print(f"\033[2m[{node_name}] attempts={updates.get('attempts', '?')}\033[0m")
                elif node_name == "retrieve":
                    no_source = sum(1 for s in updates.get("plan", []) if s.get("source") is None)
                    print(f"\033[2m[{node_name}] {no_source} step(s) have no source\033[0m")
                else:
                    print(f"\033[2m[{node_name}]\033[0m")
                for msg in updates.get("messages", [])[len(prev_messages):] if "messages" in updates else []:
                    print(f"BOT: {msg}")
                if "messages" in updates:
                    prev_messages = updates["messages"]

    print("\n--- Done! ---")


if __name__ == "__main__":
    _cli()
