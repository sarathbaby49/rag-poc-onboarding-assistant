"""REFERENCE SOLUTION — Layer 5 model selection & routing (C2 + C3).

Copy the function bodies into src/models.py.
"""

from __future__ import annotations

import re

from src import config

CHEAP_MODEL = config.CHEAP_MODEL
MID_MODEL = config.MID_MODEL
STRONG_MODEL = config.STRONG_MODEL

# Words that signal reasoning, debugging or design work.
_HARD_WORDS = (
    "why", "error", "fail", "fix", "debug", "explain", "compare", "design",
    "trace", "exception", "broken", "crash", "refactor", "suggest", "trade-off",
)
# Code-ish signals: stack traces, file:line, HTTP error codes, snippets.
_CODE_PATTERN = re.compile(r"(traceback|\.py\b|line \d+|\b[45]\d\d\b|`|\(\)|::|=>)", re.I)


def pick_model(question: str) -> str:
    q = question.lower()
    if len(q) > 150:
        return STRONG_MODEL
    if any(re.search(rf"\b{re.escape(w)}", q) for w in _HARD_WORDS):
        return STRONG_MODEL
    if _CODE_PATTERN.search(question):
        return STRONG_MODEL
    return CHEAP_MODEL


STEPS = ("parse_request", "draft_plan", "replan")


def model_for_step(step: str, attempts: int = 0) -> str:
    if step == "parse_request":
        return CHEAP_MODEL
    if step == "replan" and attempts >= 2:
        return STRONG_MODEL
    return MID_MODEL
