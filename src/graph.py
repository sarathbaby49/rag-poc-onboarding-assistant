"""Layer 4 — LangGraph guided onboarding flow (YOUR EXERCISES: G1 + G2 + G3).

A plain chatbot is reactive. Onboarding is a *process* with stages:

    welcome ──► environment setup ──► architecture tour ──► first ticket
                      │
                      ▼ (stuck?)
               mentor checkpoint

LangGraph lets you model that as a **state machine**: each node is a step,
edges decide what comes next, and "mentor checkpoints" pause the flow for a
human to confirm ("did setup actually work?") before moving on.

Concepts covered:
  - LangGraph StateGraph, nodes, edges, conditional edges
  - TypedDict state schemas
  - Human-in-the-loop (interrupt_before for mentor checkpoints)

Self-check:  python -m checks.check_graph
"""

from __future__ import annotations

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from src import config


# --- State schema (provided) -------------------------------------------------
class OnboardingState(TypedDict):
    """The state that flows through the onboarding graph.

    LangGraph passes this dict from node to node. Each node receives the current
    state and returns updates to it.
    """
    joinee: str            # the new joiner's name
    role: str              # e.g. "backend", "frontend", "fullstack"
    stage: str             # current stage: "welcome", "setup", "architecture", "first_ticket"
    messages: list[dict]   # conversation history [{"role": ..., "content": ...}]
    needs_mentor: bool     # True when the flow should pause for mentor review
    mentor_notes: str      # notes from the mentor after a checkpoint
    completed_steps: list[str]  # stages the joinee has completed


# --- G1: Implement node functions ---------------------------------------------
# Each node is a function(state) -> partial state update.
# The returned dict is MERGED into the current state (not replaced).
#
# Self-check:  python -m checks.check_graph   (section G1)
# ─────────────────────────────────────────────────────────────────────────────

def welcome(state: OnboardingState) -> dict:
    """EXERCISE G1a — Welcome node.

    Greet the joinee, confirm their role, set stage to "setup".

    Should return a dict with:
      - "stage": "setup"
      - "messages": state["messages"] + [the welcome message]
      - "completed_steps": state["completed_steps"] + ["welcome"]

    Hint: use the joinee's name and role from state to personalize the greeting.
    """
    # TODO(G1a): implement welcome node
    raise NotImplementedError("Exercise G1a: implement welcome node — see EXERCISES.md")


def setup_help(state: OnboardingState) -> dict:
    """EXERCISE G1b — Setup help node.

    Answer the joinee's setup question. If the question mentions being stuck,
    errors, or failures, set needs_mentor = True.

    Should return a dict with:
      - "messages": state["messages"] + [the answer]
      - "needs_mentor": True/False based on whether the joinee seems stuck

    Hint: use a simple keyword check for stuck detection (e.g. "error", "fail",
    "stuck", "broken", "not working"). For the answer, you can use the RAG
    pipeline (src.rag.answer) or the agent (src.agent.run_agent).
    """
    # TODO(G1b): implement setup_help node
    raise NotImplementedError("Exercise G1b: implement setup_help node — see EXERCISES.md")


def mentor_checkpoint(state: OnboardingState) -> dict:
    """EXERCISE G1c — Mentor checkpoint node.

    This node runs AFTER a human mentor has reviewed. In a real system,
    LangGraph's interrupt_before pauses execution here, a mentor adds notes,
    then the graph resumes.

    Should return a dict with:
      - "needs_mentor": False (mentor has reviewed)
      - "messages": state["messages"] + [a message noting the mentor signed off]
      - "stage": "architecture" (advance to the next stage)
      - "completed_steps": state["completed_steps"] + ["setup"]
    """
    # TODO(G1c): implement mentor_checkpoint node
    raise NotImplementedError("Exercise G1c: implement mentor_checkpoint node — see EXERCISES.md")


def architecture_tour(state: OnboardingState) -> dict:
    """EXERCISE G1d — Architecture tour node.

    Walk the joinee through the system architecture. Answer architecture
    questions using the RAG pipeline.

    Should return a dict with:
      - "stage": "first_ticket"
      - "messages": state["messages"] + [architecture overview]
      - "completed_steps": state["completed_steps"] + ["architecture"]
    """
    # TODO(G1d): implement architecture_tour node
    raise NotImplementedError("Exercise G1d: implement architecture_tour node — see EXERCISES.md")


# --- G2: Routing function -----------------------------------------------------
# Conditional edges use a routing function that inspects the state and returns
# the name of the next node to visit.
#
# Self-check:  python -m checks.check_graph   (section G2)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_setup(state: OnboardingState) -> Literal["checkpoint", "architecture"]:
    """EXERCISE G2 — Decide what comes after the setup node.

    If needs_mentor is True  → return "checkpoint" (pause for mentor review)
    If needs_mentor is False → return "architecture" (proceed to next stage)

    This is the function you'll pass to g.add_conditional_edges("setup", ...).
    """
    # TODO(G2): implement routing logic
    raise NotImplementedError("Exercise G2: implement route_after_setup — see EXERCISES.md")


# --- G3: Assemble the graph ---------------------------------------------------
# Wire the nodes and edges into a StateGraph and compile it.
#
# Self-check:  python -m checks.check_graph   (section G3)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """EXERCISE G3 — Assemble the LangGraph onboarding flow.

    Steps:
      1. Create a StateGraph(OnboardingState)
      2. Add nodes: "welcome", "setup", "checkpoint", "architecture"
      3. Set "welcome" as the entry point
      4. Add edges:
           welcome  ──►  setup
           setup    ──►  checkpoint OR architecture  (conditional, use route_after_setup)
           checkpoint ──► architecture
           architecture ──► END
      5. Compile with interrupt_before=["checkpoint"] for mentor checkpoints
      6. Return the compiled graph

    Example:
      g = StateGraph(OnboardingState)
      g.add_node("welcome", welcome)
      ...
      g.add_edge("welcome", "setup")
      g.add_conditional_edges("setup", route_after_setup)
      ...
      return g.compile(interrupt_before=["checkpoint"])
    """
    # TODO(G3): build and return the compiled graph
    raise NotImplementedError("Exercise G3: build the LangGraph flow — see EXERCISES.md")


# --- CLI entry point ----------------------------------------------------------
def _cli() -> None:
    """Quick demo: run the graph with a sample joinee."""
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Alex"
    role = sys.argv[2] if len(sys.argv) > 2 else "backend"

    graph = build_graph()
    initial_state: OnboardingState = {
        "joinee": name,
        "role": role,
        "stage": "welcome",
        "messages": [],
        "needs_mentor": False,
        "mentor_notes": "",
        "completed_steps": [],
    }

    print(f"🚀 Starting onboarding flow for {name} ({role})...\n")
    result = graph.invoke(initial_state)

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
