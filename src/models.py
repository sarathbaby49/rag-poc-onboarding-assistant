"""Layer 5 — Model selection & routing.  (EXERCISES C2 + C3)

Not every question needs the most expensive model. "What's the repo URL?" can go
to a cheap, fast model; "Why is my migration failing?" needs a strong one.
Routing is the single biggest cost lever: on our numbers, sending 60% of
questions to the cheap tier cuts the monthly bill by more than half.

The three tiers come from src/config.py (set them in .env):
    CHEAP_MODEL   claude-haiku-4-5    $1 / $5   per 1M tokens (in / out)
    MID_MODEL     claude-sonnet-5-5   $2 / $10
    STRONG_MODEL  claude-opus-5-5     $4 / $20
(Prices checked 4 Oct 2026 — see src/cost.py for the full table.)

C2 — pick_model(question): route one user question to a tier.
C3 — model_for_step(step, attempts): pick a tier per step of the LangGraph
     onboarding-plan bot ("Create a 10-day onboarding plan for a new backend
     engineer"). The Cost Lab wires your function into the graph for you.
"""

from __future__ import annotations

from src import config

CHEAP_MODEL = config.CHEAP_MODEL
MID_MODEL = config.MID_MODEL
STRONG_MODEL = config.STRONG_MODEL


def pick_model(question: str) -> str:
    """EXERCISE C2 — return the model to use for this question.

    Right now everything goes to STRONG_MODEL (safe, but the most expensive).
    Replace it with a simple rule, for example:
      - debugging / reasoning words ("why", "error", "fail", "fix", "explain",
        "compare", "debug"), stack traces or code     -> STRONG_MODEL
      - very long questions (say > 150 characters)     -> STRONG_MODEL
      - short factual lookups ("what's the repo URL?") -> CHEAP_MODEL

    Self-check: python -m checks.check_routing  (12 labelled questions, need 10+)
    Bonus: instead of rules, ask CHEAP_MODEL to label the question "easy" or
    "hard" and route on its answer. Is the extra call worth it?
    """
    return STRONG_MODEL


# The three steps of the plan bot that call a model:
#   "parse_request" — pull the role and number of days out of the request (extraction)
#   "draft_plan"    — write the day-by-day plan (needs judgment)
#   "replan"        — fix ungrounded steps or apply mentor feedback
STEPS = ("parse_request", "draft_plan", "replan")


def model_for_step(step: str, attempts: int = 0) -> str:
    """EXERCISE C3 — return the model for one step of the plan graph.

    `attempts` is how many replans have already happened (0 on the first replan).

    Target:
      - "parse_request" -> CHEAP_MODEL   (simple extraction)
      - "draft_plan"    -> MID_MODEL     (writing needs judgment)
      - "replan"        -> MID_MODEL, but escalate to STRONG_MODEL once
                           attempts >= 2 (cheaper tries already failed)

    The other nodes (retrieve, check_grounding, publish) never call a model,
    and mentor_approval is a person. Self-check: python -m checks.check_routing
    """
    return STRONG_MODEL
