"""REFERENCE SOLUTION for src/graph.py (Exercises G1 + G2 + G3).

LangGraph guided onboarding flow with mentor checkpoints.

Try the exercise first! If you're stuck or out of time, copy the relevant
function into src/graph.py.
"""

from __future__ import annotations

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from src import config


# --- State schema -------------------------------------------------------------

class OnboardingState(TypedDict):
    joinee: str
    role: str
    stage: str
    messages: list[dict]
    needs_mentor: bool
    mentor_notes: str
    completed_steps: list[str]


# --- G1: Node functions -------------------------------------------------------

def welcome(state: OnboardingState) -> dict:
    """Greet the joinee and advance to setup."""
    name = state["joinee"]
    role = state.get("role", "engineer")
    greeting = (
        f"👋 Welcome to the team, {name}! I see you're joining as a {role} engineer.\n\n"
        f"I'll guide you through onboarding step by step:\n"
        f"  1. Environment setup\n"
        f"  2. Architecture tour\n"
        f"  3. Your first ticket\n\n"
        f"Let's start with getting your development environment set up. "
        f"Ask me anything about the setup process!"
    )
    return {
        "stage": "setup",
        "messages": state["messages"] + [{"role": "assistant", "content": greeting}],
        "completed_steps": state["completed_steps"] + ["welcome"],
    }


def setup_help(state: OnboardingState) -> dict:
    """Answer setup questions; flag for mentor if the joinee seems stuck."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        answer_text = (
            "To set up your environment, run these commands:\n\n"
            "```bash\n"
            "python3 -m venv .venv && source .venv/bin/activate\n"
            "pip install --upgrade pip\n"
            "pip install -r requirements.txt\n"
            "python -m src.ingest\n"
            "```\n\n"
            "Let me know if you hit any issues!"
        )
        needs_mentor = False
    else:
        stuck_keywords = {"error", "fail", "stuck", "broken", "not working", "crash", "exception", "traceback"}
        needs_mentor = any(kw in last_user_msg.lower() for kw in stuck_keywords)

        if needs_mentor:
            answer_text = (
                "It sounds like you're hitting a setup issue. Let me flag this for a mentor "
                "to review. In the meantime, here are some common fixes:\n\n"
                "1. Make sure you're using Python 3.10+: `python3 --version`\n"
                "2. Try recreating the venv: `rm -rf .venv && python3 -m venv .venv`\n"
                "3. Check that all system dependencies are installed\n\n"
                "A mentor will review and sign off before we continue."
            )
        else:
            answer_text = (
                f"Regarding your question: '{last_user_msg[:80]}'\n\n"
                "I'd recommend checking the setup docs. Run `python explore.py stats` "
                "to verify your index is built correctly. Let me know if anything looks off!"
            )

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": answer_text}],
        "needs_mentor": needs_mentor,
    }


def mentor_checkpoint(state: OnboardingState) -> dict:
    """Mentor has reviewed — acknowledge and advance."""
    mentor_notes = state.get("mentor_notes", "Approved by mentor.")
    msg = (
        f"✅ Mentor review complete: {mentor_notes}\n\n"
        f"Great, your setup is confirmed working! Let's move on to "
        f"understanding the system architecture."
    )
    return {
        "needs_mentor": False,
        "messages": state["messages"] + [{"role": "assistant", "content": msg}],
        "stage": "architecture",
        "completed_steps": state["completed_steps"] + ["setup"],
    }


def architecture_tour(state: OnboardingState) -> dict:
    """Walk the joinee through the architecture."""
    overview = (
        "🏗️ **Architecture Overview — Acme Shop**\n\n"
        "The system is an e-commerce backend with these main components:\n\n"
        "- **Payment Service** (`code/payment_providers.py`): Handles payment processing "
        "with Razorpay and Stripe providers.\n"
        "- **Auth Service** (`code/auth.py`): JWT-based authentication with refresh tokens.\n"
        "- **Data Layer**: PostgreSQL for transactions, Redis for sessions.\n\n"
        "The onboarding docs in `data/sample_company/` cover each piece. "
        "Use `python explore.py query \"...\"` to search for specific topics.\n\n"
        "Next up: your first ticket! 🎫"
    )
    return {
        "stage": "first_ticket",
        "messages": state["messages"] + [{"role": "assistant", "content": overview}],
        "completed_steps": state["completed_steps"] + ["architecture"],
    }


# --- G2: Routing function -----------------------------------------------------

def route_after_setup(state: OnboardingState) -> Literal["checkpoint", "architecture"]:
    """Route to mentor checkpoint if stuck, otherwise continue."""
    if state.get("needs_mentor", False):
        return "checkpoint"
    return "architecture"


# --- G3: Build the graph ------------------------------------------------------

def build_graph():
    """Assemble and compile the LangGraph onboarding flow."""
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


# --- CLI ----------------------------------------------------------------------

def run_onboarding(name: str, role: str, user_message: str = "") -> dict:
    """Run the onboarding flow."""
    graph = build_graph()
    initial_state: OnboardingState = {
        "joinee": name,
        "role": role,
        "stage": "welcome",
        "messages": [{"role": "user", "content": user_message}] if user_message else [],
        "needs_mentor": False,
        "mentor_notes": "",
        "completed_steps": [],
    }
    return graph.invoke(initial_state)


def _cli() -> None:
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Alex"
    role = sys.argv[2] if len(sys.argv) > 2 else "backend"

    print(f"🚀 Starting onboarding flow for {name} ({role})...\n")
    result = run_onboarding(name, role)

    print("\n--- Final state ---")
    print(f"Stage: {result['stage']}")
    print(f"Completed: {result['completed_steps']}")
    print(f"Messages: {len(result['messages'])} messages")
    for msg in result["messages"]:
        role_label = msg.get("role", "?")
        content = msg.get("content", "")[:120]
        print(f"  [{role_label}] {content}...")


if __name__ == "__main__":
    _cli()
