"""REFERENCE SOLUTION for src/graph.py (Exercises G1 + G2 + G3).

LangGraph "Onboarding Plan Chatbot" — a new hire asks for a 2-week
onboarding plan. The graph plans, grounds every step in retrieved docs
(RAG-lite), loops back to re-plan when a step has no supporting doc, then
pauses for a mentor to approve or reject before publishing.

    START → plan → retrieve → check_grounding ─┬─► mentor_approval ─┬─► publish → END
                                                │                    │
                                                └─► replan ──────────┘
                                                   (gaps / rejected)

Try the exercise first! If you're stuck or out of time, copy the relevant
function into src/graph.py.
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
# Constants
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3  # Safety valve: prevents infinite replan loops.

DOCS_DIR = Path(config.BASE_DIR) / "data" / "sample_company"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanState(TypedDict):
    user_request: str
    role: str
    days: int
    plan: list[dict]          # [{"day": 1, "task": "...", "source": "..." | None}]
    gaps: list[str]           # tasks with no supporting doc
    mentor_feedback: str
    approved: bool
    attempts: int
    messages: list[str]       # bot messages shown to the user


# ---------------------------------------------------------------------------
# Helpers
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
    """Return the first doc filename whose content matches keywords from *task*."""
    words = set(re.findall(r"[a-z]{3,}", task.lower()))
    # Discard generic stop-words that would match everything
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
# LLM helpers
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
# Default (deterministic) planner
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
    """Build a day-by-day onboarding plan (deterministic fallback)."""
    tasks = _TEMPLATE_PLAN[:days]
    return [{"day": i + 1, "task": t, "source": None} for i, t in enumerate(tasks)]


def _llm_plan(role: str, days: int, doc_names: list[str]) -> list[dict]:
    """Use the LLM to generate a role-specific onboarding plan.

    The prompt asks the LLM to return JSON. Falls back to the deterministic
    template if parsing fails.
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
        # Extract JSON array from response (handle markdown fences)
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
) -> list[dict]:
    """Use the LLM to fix un-grounded steps and apply mentor feedback.

    Falls back to rule-based replacement if parsing fails.
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
    return None  # signal fallback to rule-based


# ---------------------------------------------------------------------------
# G1  Node functions
# ---------------------------------------------------------------------------

def plan(state: PlanState) -> dict:
    """Parse the request, use LLM to draft a role-specific plan, greet the user."""
    role, days = _parse_request(state["user_request"])
    doc_names = list(_load_docs().keys())
    draft = _llm_plan(role, days, doc_names)
    return {
        "role": role,
        "days": days,
        "plan": draft,
        "messages": state["messages"] + [
            f"👋 Welcome! I'll build a {days}-day onboarding plan for a **{role}** developer. Give me a moment…"
        ],
    }


def retrieve(state: PlanState) -> dict:
    """Ground each plan step: find a matching doc or leave source as None."""
    docs = _load_docs()
    updated = []
    for step in state["plan"]:
        source = _keyword_search(step["task"], docs)
        updated.append({**step, "source": source})
    return {"plan": updated}


def check_grounding(state: PlanState) -> dict:
    """Identify un-grounded steps (source is None)."""
    gaps = [
        f"Day {s['day']}: {s['task']}"
        for s in state["plan"]
        if s["source"] is None
    ]
    return {"gaps": gaps}


def replan(state: PlanState) -> dict:
    """Use LLM to replace un-grounded steps and apply mentor feedback.

    Falls back to rule-based replacement if the LLM call fails.
    """
    docs = _load_docs()
    doc_names = list(docs.keys())
    feedback = state.get("mentor_feedback", "")

    # Try LLM-based replan
    llm_result = _llm_replan(state["plan"], state["gaps"], feedback, doc_names)

    if llm_result is not None:
        new_plan = llm_result
    else:
        # Fallback: rule-based replacement
        _REPLACEMENTS: dict[str, str] = {
            "complete mandatory compliance training": "Read the CI/CD pipeline overview",
        }
        new_plan = []
        for step in state["plan"]:
            task_lower = step["task"].lower()
            if step["source"] is None:
                replacement = _REPLACEMENTS.get(task_lower, f"Review {doc_names[0] if doc_names else 'docs'}")
                new_plan.append({**step, "task": replacement, "source": None})
            else:
                new_plan.append(step)

        if feedback:
            day_match = re.search(r"day\s*(\d+)", feedback, re.IGNORECASE)
            if day_match:
                target_day = int(day_match.group(1))
                action = feedback.strip()
                for i, step in enumerate(new_plan):
                    if step["day"] == target_day:
                        new_plan[i] = {**step, "task": action, "source": None}
                        break
                else:
                    new_plan.append({"day": target_day, "task": action, "source": None})
                    new_plan.sort(key=lambda s: s["day"])

    return {
        "plan": new_plan,
        "gaps": [],
        "mentor_feedback": "",
        "attempts": state["attempts"] + 1,
    }


def mentor_approval(state: PlanState) -> dict:
    """Pause for mentor approval using LangGraph interrupt."""
    from langgraph.types import interrupt

    plan_text = "\n".join(
        f"  Day {s['day']}: {s['task']}  ({s['source'] or '?'})"
        for s in state["plan"]
    )
    mentor_response = interrupt({
        "question": "Please review and approve this onboarding plan, or give feedback.",
        "plan": plan_text,
    })

    if isinstance(mentor_response, str) and mentor_response.strip().lower() == "approve":
        return {
            "approved": True,
            "messages": state["messages"] + [
                "✅ Mentor **approved** the plan!"
            ],
        }
    else:
        return {
            "approved": False,
            "mentor_feedback": str(mentor_response),
            "messages": state["messages"] + [
                f"📝 Mentor feedback: _{mentor_response}_ — re-planning…"
            ],
        }


def publish(state: PlanState) -> dict:
    """Format and publish the final approved plan."""
    lines = [f"📋 **Your {state['days']}-day onboarding plan:**\n"]
    for s in state["plan"]:
        src = s["source"] or "—"
        lines.append(f"- **Day {s['day']}:** {s['task']}  _({src})_")
    lines.append("\n🎉 You're all set — good luck!")
    return {
        "messages": state["messages"] + ["\n".join(lines)],
    }


# ---------------------------------------------------------------------------
# G2  Routing functions
# ---------------------------------------------------------------------------

def route_after_check(state: PlanState) -> str:
    """Route after check_grounding: replan if gaps remain, else mentor."""
    if state["gaps"] and state["attempts"] < MAX_ATTEMPTS:
        return "replan"
    return "mentor_approval"


def route_after_mentor(state: PlanState) -> str:
    """Route after mentor_approval: publish if approved, else replan."""
    if state["approved"]:
        return "publish"
    return "replan"


# ---------------------------------------------------------------------------
# G3  Build the graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Assemble the onboarding-plan chatbot graph."""
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


# ---------------------------------------------------------------------------
# Helpers for UI / CLI
# ---------------------------------------------------------------------------

def run_graph_turn(
    user_message: str,
    graph_state: dict | None = None,
    thread_id: str = "default",
    checkpointer=None,
) -> tuple[dict, list[str], bool]:
    """Run one turn of the plan chatbot.

    Returns (new_graph_state_dict, new_bot_messages, is_waiting_for_mentor).
    graph_state_dict holds the opaque state needed to resume.
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
        # Graph already finished
        return graph_state, ["The plan has already been published! Clear the conversation to start over."], False

    # Collect the final state snapshot
    snapshot = graph.get_state(config)
    final_state = dict(snapshot.values) if snapshot.values else {}

    all_msgs = final_state.get("messages", [])
    new_msgs = all_msgs[len(prev_msgs):]

    # Check if interrupted (waiting for mentor)
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
    """Interactive terminal chatbot — matches the sample-run in the spec."""
    from langgraph.types import Command

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "cli"}}

    request = input("You: ").strip()
    if not request:
        request = "Hi, I'm joining Monday as a mid-level frontend dev. Can you plan my first 2 weeks?"

    initial = PlanState(
        user_request=request,
        role="",
        days=10,
        plan=[],
        gaps=[],
        mentor_feedback="",
        approved=False,
        attempts=0,
        messages=[],
    )

    prev_messages: list[str] = []

    # Stream until interrupt
    for event in graph.stream(initial, config, stream_mode="updates"):
        for node_name, updates in event.items():
            # Trace line
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

            # Print new bot messages
            for msg in updates.get("messages", [])[len(prev_messages):] if "messages" in updates else []:
                print(f"BOT: {msg}")
            if "messages" in updates:
                prev_messages = updates["messages"]

    # Mentor loop
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        state = snapshot.values
        # Show plan
        print("\n--- Plan for mentor review ---")
        for s in state.get("plan", []):
            print(f"  Day {s['day']}: {s['task']}  ({s['source'] or '?'})")
        mentor_input = input("\nMENTOR (Sarath) — type 'approve' or give feedback: ").strip()

        for event in graph.stream(Command(resume=mentor_input), config, stream_mode="updates"):
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

    print("\n--- Done! ---")


if __name__ == "__main__":
    _cli()
