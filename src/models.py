"""Layer 5 — Model selection & routing (STUB / your mission).

Not every question needs the most expensive model. "What's the repo URL?" can go
to a cheap fast model; "Why is my migration failing?" needs a strong one.
Routing simple questions to a cheaper model is one of the biggest cost levers.

Model menu (see the pricing table in the Anthropic docs):
    claude-opus-5    — strongest, most expensive   ($5 / $25 per 1M tok)
    claude-sonnet-5  — balanced                     ($2 / $10)
    claude-haiku-4-5 — cheapest, fastest            ($1 / $5)

Your mission:
  1. Implement `pick_model` with a simple heuristic (length, keywords) or a
     tiny classifier call to a cheap model.
  2. Wire it into src.rag.answer so the model is chosen per question.
  3. (Bonus) Measure cost per question before/after to prove the saving.
"""

from __future__ import annotations

from src import config

CHEAP_MODEL = "claude-haiku-4-5"
STRONG_MODEL = config.ANSWER_MODEL  # claude-opus-5


def pick_model(question: str) -> str:
    """Return the model id to use for this question.

    TODO: replace this always-strong default with a real routing rule, e.g.
      - short, factual lookups        -> CHEAP_MODEL
      - "why", "debug", "error", code -> STRONG_MODEL
    """
    return STRONG_MODEL
