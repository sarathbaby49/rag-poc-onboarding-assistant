"""Self-check for Lab 1 — the Evaluation exercises (EV1–EV4).

Key-free (always run):  EV1 grading, EV2 retrieval hit-rate, EV3 JSON parsing
with a fake judge, EV4 regression gate on two hand-made runs.
Live (needs the gateway): EV3 against a real judge call.

Run from the repo root:  python -m checks.check_eval
(Presenters: `python -m checks.check_eval --solutions` checks the reference answers.)
"""

from __future__ import annotations

import sys

from src import layer6_support

if "--solutions" in sys.argv:
    layer6_support.use_reference_solutions()

from src.eval import run_eval  # noqa: E402  (after the optional swap)
from src.rag import ABSTAIN_MESSAGE  # noqa: E402


def _check(label: str, cond) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return bool(cond)


def _todo(code: str) -> bool:
    print(f"❌ {code} still a TODO — see layer6-evaluation-and-production/EXERCISES.md")
    return False


GOLDEN = {"id": "pay-1", "question": "What payment providers do we use?",
          "expected_points": ["Stripe", "PayPal", "Razorpay"],
          "expected_source": "data/sample_company/payments.md"}
PAYMENTS_SRC = [{"source": "data/sample_company/payments.md", "text": "...", "score": 0.6}]
OFF_TOPIC = {"id": "off-2", "question": "What is the capital of Australia?",
             "expect_abstain": True, "must_not_contain": ["Canberra"]}


def main() -> None:
    results: list[bool] = []

    print("-- Implementation status --")
    for code, name, done in layer6_support.exercise_status():
        if code.startswith("EV"):
            print(f"{'✅' if done else '❌'} {code} {name}() — {'implemented' if done else 'NOT implemented yet'}")

    print("\n-- EV1: grade_keywords (no key) --")
    try:
        g = run_eval.grade_keywords
        full = g({"answer": "We use Stripe, PayPal and Razorpay [1].", "sources": PAYMENTS_SRC}, GOLDEN)
        results.append(_check("all points + right source -> passed, coverage 1.0",
                              full["passed"] and abs(full["coverage"] - 1.0) < 1e-9))
        partial = g({"answer": "Stripe only.", "sources": PAYMENTS_SRC}, GOLDEN)
        results.append(_check("1 of 3 points -> coverage ≈ 0.33, not passed",
                              not partial["passed"] and abs(partial["coverage"] - 1 / 3) < 0.01))
        uncited = g({"answer": "Stripe, PayPal, Razorpay.",
                     "sources": [{"source": "data/sample_company/setup.md"}]}, GOLDEN)
        results.append(_check("right words but wrong source -> not passed (cited=False)",
                              not uncited["passed"] and uncited["cited"] is False))
        results.append(_check("off-topic + abstain message -> passed",
                              g({"answer": ABSTAIN_MESSAGE, "sources": []}, OFF_TOPIC)["passed"]))
        leaked = g({"answer": "The docs don't cover that, but it's Canberra.", "sources": []}, OFF_TOPIC)
        results.append(_check("'not in the docs, but it's Canberra' -> NOT passed (must_not_contain)",
                              not leaked["passed"]))
    except NotImplementedError:
        results.append(_todo("EV1"))

    print("\n-- EV2: retrieval_hit (no key, uses the local index) --")
    try:
        results.append(_check("payments question retrieves payments.md",
                              run_eval.retrieval_hit(GOLDEN) is True))
        results.append(_check("off-topic case (no expected_source) -> None",
                              run_eval.retrieval_hit(OFF_TOPIC) is None))
    except NotImplementedError:
        results.append(_todo("EV2"))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"retrieval_hit runs ({type(e).__name__}: {e}) — built the index? "
                              "python -m src.ingest", False))

    print("\n-- EV3: judge_faithfulness (parsing, no key) --")
    try:
        abstained = run_eval.judge_faithfulness("q", {"abstained": True, "answer": ABSTAIN_MESSAGE, "sources": []})
        results.append(_check("abstained answer -> PASS without calling the model",
                              abstained["verdict"] == "PASS"))
        home = sys.modules[run_eval.judge_faithfulness.__module__]   # src or solutions
        original = home.complete
        try:
            home.complete = lambda *a, **k: 'Grounded in [1].\n{"verdict": "pass", "reasoning": "ok"}'
            ok = run_eval.judge_faithfulness("q", {"abstained": False, "answer": "a", "sources": PAYMENTS_SRC})
            results.append(_check("reads the JSON on the last line (and upper-cases the verdict)",
                                  ok["verdict"] == "PASS"))
            home.complete = lambda *a, **k: "I think it's fine."
            bad = run_eval.judge_faithfulness("q", {"abstained": False, "answer": "a", "sources": PAYMENTS_SRC})
            results.append(_check("no JSON from the judge -> FAIL (never crash)", bad["verdict"] == "FAIL"))
        finally:
            home.complete = original
    except NotImplementedError:
        results.append(_todo("EV3"))

    print("\n-- EV4: compare_runs (no key) --")
    try:
        base = {"rates": {"answer_pass": 0.9, "faithful": 1.0, "retrieval_hit": None},
                "cases": [{"id": "a", "checks": {"answer_pass": True, "faithful": True}},
                          {"id": "b", "checks": {"answer_pass": False, "faithful": True}}]}
        same = run_eval.compare_runs(base, base)
        results.append(_check("identical runs -> SHIP, no flips", same["verdict"] == "SHIP" and not same["flips"]))
        worse = {"rates": {"answer_pass": 0.9, "faithful": 0.5, "retrieval_hit": None},
                 "cases": [{"id": "a", "checks": {"answer_pass": True, "faithful": False}},
                           {"id": "b", "checks": {"answer_pass": False, "faithful": True}}]}
        res = run_eval.compare_runs(base, worse)
        results.append(_check("faithfulness drop -> BLOCK, regression listed",
                              res["verdict"] == "BLOCK" and "faithful" in res["regressions"]))
        results.append(_check("case 'a' flagged as a pass→fail flip on faithful",
                              {"id": "a", "metric": "faithful"} in res["flips"]))
        results.append(_check("None rates are skipped in deltas", "retrieval_hit" not in res["deltas"]))
    except NotImplementedError:
        results.append(_todo("EV4"))

    print("\n-- EV3 live: a real judge call (needs the gateway) --")
    ok, msg = layer6_support.gateway_reachable()
    if not ok:
        print(f"⚠️  skipped: {msg}")
    else:
        try:
            verdict = run_eval.judge_faithfulness(
                "What payment providers do we use?",
                {"abstained": False, "answer": "We use Stripe and also Bitcoin [1].",
                 "sources": [{"source": "payments.md", "text": "We use Stripe, PayPal and Razorpay."}]},
            )
            print(f"{'✅' if verdict['verdict'] == 'FAIL' else '⚠️ '} judge flags an invented claim "
                  f"(Bitcoin) as {verdict['verdict']}: {verdict['reasoning']}")
        except NotImplementedError:
            print("⚠️  EV3 not implemented yet")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  judge call failed ({type(e).__name__}: {e})")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} key-free checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
