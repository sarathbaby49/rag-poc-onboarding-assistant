"""Layer 4 — LangGraph guided onboarding flow (STUB / your mission).

A plain chatbot is reactive. Onboarding is a *process* with stages:

    welcome -> environment setup -> architecture tour -> first ticket -> first PR

LangGraph lets you model that as a state machine: each node is a step, edges
decide what comes next, and "mentor checkpoints" pause the flow for a human to
confirm ("did setup actually work?") before moving on.

This stub sketches the nodes and state. Fill in the node functions and wire the
graph to complete Layer 4.
"""

from __future__ import annotations

from typing import TypedDict

# from langgraph.graph import StateGraph, END   # uncomment when you implement


class OnboardingState(TypedDict):
    joinee: str
    stage: str
    question: str
    answer: str
    needs_mentor: bool


def welcome(state: OnboardingState) -> OnboardingState:
    # TODO: greet, confirm role, set stage = "setup".
    raise NotImplementedError


def setup_help(state: OnboardingState) -> OnboardingState:
    # TODO: answer setup questions (reuse src.rag.answer), detect when stuck,
    # and set needs_mentor = True to trigger a checkpoint.
    raise NotImplementedError


def mentor_checkpoint(state: OnboardingState) -> OnboardingState:
    # TODO: pause for a human mentor to sign off before advancing.
    raise NotImplementedError


def build_graph():
    """TODO: assemble the StateGraph.

        g = StateGraph(OnboardingState)
        g.add_node("welcome", welcome)
        g.add_node("setup", setup_help)
        g.add_node("checkpoint", mentor_checkpoint)
        g.add_edge("welcome", "setup")
        g.add_conditional_edges("setup", route_after_setup)
        ...
        return g.compile()
    """
    raise NotImplementedError("Your mission: build the LangGraph onboarding flow.")
