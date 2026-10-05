"""Self-check for Layer 5 exercises C2 (pick_model) and C3 (model_for_step).

Pure functions — NO key needed. Run from the repo root:
    python -m checks.check_routing
"""

from __future__ import annotations

import sys

from src.cost import CHEAP_MODEL, MID_MODEL, STRONG_MODEL, model_for_step, pick_model

# Labelled questions: (question, expected tier)
QUESTIONS = [
    ("What's the repo URL?", CHEAP_MODEL),
    ("Which port does the API run on?", CHEAP_MODEL),
    ("Who owns the payments service?", CHEAP_MODEL),
    ("Where is the CI config?", CHEAP_MODEL),
    ("What Python version do we use?", CHEAP_MODEL),
    ("List the payment providers we support.", CHEAP_MODEL),
    ("Why does my database migration fail with a foreign key error?", STRONG_MODEL),
    ("Tests pass locally but fail in CI. How do I debug that?", STRONG_MODEL),
    ("Explain how refresh tokens work in our auth flow and the security trade-offs.", STRONG_MODEL),
    ("My Razorpay webhook returns 401, how do I fix it?", STRONG_MODEL),
    ("Compare our Stripe and Razorpay integrations and suggest how to add a third provider.", STRONG_MODEL),
    ("KeyError 'user_id' in auth.py line 42 when I log in — what's wrong?", STRONG_MODEL),
]
NEEDED = 10


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def _tier(model: str) -> str:
    return {CHEAP_MODEL: "cheap", MID_MODEL: "mid", STRONG_MODEL: "strong"}.get(model, model)


def check_c2() -> bool:
    print("C2 · pick_model\n")
    correct = 0
    for q, expected in QUESTIONS:
        got = pick_model(q)
        ok = got == expected
        correct += ok
        print(f"  {'✓' if ok else '✗'} {_tier(got):>6}  (want {_tier(expected):>6})  {q}")
    cheap_share = sum(pick_model(q) == CHEAP_MODEL for q, _ in QUESTIONS) / len(QUESTIONS)
    print(f"\n  {correct}/{len(QUESTIONS)} routed correctly · {cheap_share:.0%} sent to the cheap tier")
    return _check(f"C2 routes at least {NEEDED}/{len(QUESTIONS)} questions correctly", correct >= NEEDED)


def check_c3() -> bool:
    print("\nC3 · model_for_step\n")
    results = [
        _check("parse_request -> cheap", model_for_step("parse_request") == CHEAP_MODEL),
        _check("draft_plan -> mid", model_for_step("draft_plan") == MID_MODEL),
        _check("replan, 1st try (attempts=0) -> mid", model_for_step("replan", 0) == MID_MODEL),
        _check("replan, 2nd try (attempts=1) -> mid", model_for_step("replan", 1) == MID_MODEL),
        _check("replan, 3rd try (attempts=2) -> strong", model_for_step("replan", 2) == STRONG_MODEL),
    ]
    return all(results)


def main() -> int:
    if len({CHEAP_MODEL, MID_MODEL, STRONG_MODEL}) < 3:
        print("⚠️  CHEAP_MODEL, MID_MODEL and STRONG_MODEL must be three different names (.env)")
        return 1
    ok_c2 = check_c2()
    ok_c3 = check_c3()
    return 0 if (ok_c2 and ok_c3) else 1


if __name__ == "__main__":
    sys.exit(main())
