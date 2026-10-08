"""Layer 6 — Evaluation  (LAB 1 — YOUR EXERCISES: EV1–EV4).

"It worked when I tried it" is not evaluation. A golden set is a fixed list of
real questions + what a good answer must contain. You run the assistant against
it every time you change something, and you watch the scores.

    golden_set.jsonl  ->  ask_assistant()  ->  graders  ->  scoreboard  ->  SHIP / BLOCK

  - EV1 (core):    implement `grade_keywords`      -> code-based grader (cheap, exact)
  - EV2 (core):    implement `retrieval_hit`       -> was the right doc retrieved? (no key)
  - EV3 (core):    implement `judge_faithfulness`  -> LLM-as-judge: is it grounded?
  - EV4 (core):    implement `compare_runs`        -> regression gate: SHIP or BLOCK

Done for you: the golden set, the system under test (src/assistant.py, prompt
versions v1/v2), the judge prompt, run_suite, and saving/loading runs.

Reference answers: solutions/eval.py
Self-check:        python -m checks.check_eval
Run it:            python -m src.eval.run_eval --prompt v1
                   python -m src.eval.run_eval --prompt v2
                   python -m src.eval.run_eval --compare v1 v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

from src import config
from src.assistant import ask_assistant
from src.llm import complete
from src.rag import ABSTAIN_MESSAGE, format_context
from src.rag_helper import semantic_search

GOLDEN = Path(__file__).parent / "golden_set.jsonl"

# An answer "covers" a golden case when it mentions at least this share of the
# expected points AND the expected source was among the sources.
PASS_COVERAGE = 0.6

# Phrases that mean "I don't know / I won't". Used for off-topic and adversarial
# cases, where declining IS the correct behaviour.
ABSTAIN_PHRASES = (
    "couldn't find", "could not find", "can't find", "cannot find", "not in the",
    "doesn't contain", "does not contain", "don't have", "do not have", "not able to",
    "can't help", "cannot help", "can't share", "cannot share", "can't provide",
    "cannot provide", "not available", "no information",
)

JUDGE_PROMPT = """You are grading one answer from an engineering onboarding assistant.

Criterion: FAITHFUL — every factual claim in the answer is supported by the
numbered context sources. Saying the information isn't available is faithful.
Facts from general knowledge that are NOT in the sources make it unfaithful.

Context sources:
{context}

Question: {question}

Answer to grade:
{answer}

Explain your reasoning in one or two sentences, then put this JSON on the last line:
{{"verdict": "PASS" or "FAIL", "reasoning": "<one sentence>"}}"""


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def is_abstention(text: str) -> bool:
    """True if the answer declines (the abstain message, or an 'I can't find it')."""
    lowered = text.lower()
    return text.strip() == ABSTAIN_MESSAGE or any(p in lowered for p in ABSTAIN_PHRASES)


# --- EV1: code-based grader ---------------------------------------------------
def grade_keywords(response: dict, case: dict) -> dict:
    """EXERCISE EV1 — the cheap, deterministic grader. Return:

        {"coverage": float 0..1, "cited": bool, "passed": bool}

    `response` is what ask_assistant() returns: {"answer", "sources", ...}.

    Two kinds of case:
      - case.get("expect_abstain") is True (off-topic / adversarial questions):
          passed = is_abstention(response["answer"]) AND none of the strings in
                   case.get("must_not_contain", []) appear in it (case-insensitive)
                   ("the docs don't cover it, but it's Canberra" is NOT an abstention)
          coverage = 1.0 if passed else 0.0;  cited = True (nothing to cite)
      - otherwise (golden questions):
          coverage = share of case["expected_points"] found in the answer
                     (case-insensitive substring match)
          cited    = case["expected_source"] appears in any source's "source"
          passed   = coverage >= PASS_COVERAGE and cited

    Self-check:  python -m checks.check_eval   (no key needed)
    """
    raise NotImplementedError("Exercise EV1: implement grade_keywords — see layer6 EXERCISES.md")


# --- EV2: retrieval metric ----------------------------------------------------
def retrieval_hit(case: dict, k: int = config.TOP_K) -> bool | None:
    """EXERCISE EV2 — score retrieval on its own (a hit-rate / recall@k check).

    A bad answer has two possible causes: retrieval fetched the wrong chunks, or
    generation misused good ones. Score them separately and you know which to fix.

    Steps:
      1. If the case has no "expected_source" (off-topic cases), return None.
      2. hits = semantic_search(case["question"], k)
      3. Return True if case["expected_source"] is in any hit's "source".

    No model call, so this runs with NO key.
    Self-check:  python -m checks.check_eval
    """
    raise NotImplementedError("Exercise EV2: implement retrieval_hit — see layer6 EXERCISES.md")


# --- EV3: LLM-as-judge --------------------------------------------------------
def judge_faithfulness(question: str, response: dict) -> dict:
    """EXERCISE EV3 — ask a model whether the answer is grounded in its sources.

    Return {"verdict": "PASS" | "FAIL", "reasoning": str}.

    Steps:
      1. If response["abstained"] is True there is nothing to check:
             return {"verdict": "PASS", "reasoning": "abstained without calling the model"}
      2. prompt = JUDGE_PROMPT.format(context=format_context(response["sources"]),
                                      question=question, answer=response["answer"])
      3. text = complete([{"role": "user", "content": prompt}],
                         model=config.JUDGE_MODEL, max_tokens=400,
                         timeout=config.LLM_TIMEOUT)
      4. Find the LAST {...} in text (re.findall(r"\\{.*?\\}", text, re.DOTALL)[-1]),
         json.loads it. If there is none or it doesn't parse, return
         {"verdict": "FAIL", "reasoning": "judge returned no valid JSON"}.
      5. Return {"verdict": data["verdict"].upper(), "reasoning": data.get("reasoning", "")}.

    Self-check:  python -m checks.check_eval   (parsing checked with no key; live part needs the gateway)
    """
    raise NotImplementedError("Exercise EV3: implement judge_faithfulness — see layer6 EXERCISES.md")


# --- EV4: regression gate -----------------------------------------------------
def compare_runs(baseline: dict, candidate: dict, tolerance: float = 0.0) -> dict:
    """EXERCISE EV4 — decide whether the candidate prompt can ship.

    A run (from run_suite) looks like:
        {"prompt_version": "v1",
         "rates": {"answer_pass": 0.92, "retrieval_hit": 1.0, "faithful": 1.0},
         "cases": [{"id": "pay-1", "checks": {"answer_pass": True, "faithful": True, ...}}, ...]}
    A rate or a check can be None (not applicable / not run) — skip those.

    Return:
        {"deltas":      {metric: candidate_rate - baseline_rate},    # both not None
         "regressions": [metrics whose delta < -tolerance],
         "flips":       [{"id": case_id, "metric": m}, ...]  # True in baseline, False in candidate
         "verdict":     "BLOCK" if regressions or flips else "SHIP"}

    Why flips too: an average can stay flat while important cases swap from pass
    to fail. Compare per case, not only the averages.

    Self-check:  python -m checks.check_eval   (no key needed)
    """
    raise NotImplementedError("Exercise EV4: implement compare_runs — see layer6 EXERCISES.md")


# --- Done for you: running the suite ------------------------------------------
METRICS = ("answer_pass", "retrieval_hit", "faithful")


def _try(fn: Callable, *args):
    """Run a grader; an unfinished exercise counts as 'not run' (None)."""
    try:
        return fn(*args)
    except NotImplementedError:
        return None


def run_suite(prompt_version: str = "v1", use_judge: bool = True,
              progress: Callable[[int, int, dict], None] | None = None) -> dict:
    """Ask every golden question, grade every answer, return the run."""
    cases = load_golden()
    results = []
    for i, case in enumerate(cases, start=1):
        response = ask_assistant(case["question"], prompt_version=prompt_version, user_id="eval-runner")
        keywords = _try(grade_keywords, response, case)
        judge = _try(judge_faithfulness, case["question"], response) if use_judge else None
        result = {
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "answer": response["answer"],
            "sources": sorted({h["source"] for h in response["sources"]}),
            "abstained": response["abstained"],
            "cost_usd": (response["usage"] or {}).get("cost_usd", 0.0),
            "judge_reasoning": judge["reasoning"] if judge else None,
            "checks": {
                "answer_pass": keywords["passed"] if keywords else None,
                "retrieval_hit": _try(retrieval_hit, case),
                "faithful": (judge["verdict"] == "PASS") if judge else None,
            },
        }
        results.append(result)
        if progress:
            progress(i, len(cases), result)

    rates = {}
    for metric in METRICS:
        values = [r["checks"][metric] for r in results if r["checks"][metric] is not None]
        rates[metric] = sum(values) / len(values) if values else None
    return {"prompt_version": prompt_version, "rates": rates, "cases": results,
            "total_cost_usd": sum(r["cost_usd"] for r in results)}


def save_run(run: dict) -> Path:
    config.EVAL_RUNS_DIR.mkdir(exist_ok=True)
    path = config.EVAL_RUNS_DIR / f"{run['prompt_version']}.json"
    path.write_text(json.dumps(run, indent=2))
    return path


def load_run(prompt_version: str) -> dict | None:
    path = config.EVAL_RUNS_DIR / f"{prompt_version}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(rate: float | None) -> str:
    return "  n/a" if rate is None else f"{rate * 100:4.0f}%"


def _print_run(run: dict) -> None:
    for r in run["cases"]:
        marks = " ".join(
            f"{m}={'—' if v is None else ('✅' if v else '❌')}" for m, v in r["checks"].items()
        )
        print(f"[{r['id']:<7}] {marks}  {r['question']}")
        if r["checks"]["faithful"] is False and r["judge_reasoning"]:
            print(f"           judge: {r['judge_reasoning']}")
    print(f"\nScoreboard for prompt {run['prompt_version']}  (cost ${run['total_cost_usd']:.4f})")
    for metric, rate in run["rates"].items():
        print(f"  {metric:<14} {_fmt(rate)}")


def _use_solutions_if_asked(names: tuple[str, ...], solution_module: str) -> None:
    """`--solutions` on the CLI: run the reference answers (presenter demo)."""
    if "--solutions" not in sys.argv:
        return
    import importlib

    from src.layer6_support import use_reference_solutions

    use_reference_solutions()
    ref = importlib.import_module(solution_module)
    globals().update({name: getattr(ref, name) for name in names})


def _require_gateway() -> None:
    from src.layer6_support import gateway_reachable

    ok, msg = gateway_reachable()
    if not ok:
        sys.exit(f"❌ {msg}\n   Saved runs in .eval_runs/ can still be compared offline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set eval.")
    parser.add_argument("--prompt", default="v1", help="prompt version to evaluate (v1, v2)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (faster, cheaper)")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="compare two saved runs (runs any that are missing)")
    parser.add_argument("--solutions", action="store_true", help="use the reference solutions (presenters)")
    args = parser.parse_args()
    _use_solutions_if_asked(("grade_keywords", "retrieval_hit", "judge_faithfulness", "compare_runs"),
                            "solutions.eval")

    if args.compare:
        runs = []
        for version in args.compare:
            run = load_run(version)
            if run is None:
                _require_gateway()
                print(f"No saved run for {version} — running it now…")
                run = run_suite(version, use_judge=not args.no_judge)
                save_run(run)
            runs.append(run)
        result = compare_runs(*runs)
        print(f"{'metric':<14} {args.compare[0]:>6} {args.compare[1]:>6}   delta")
        for metric in METRICS:
            a, b = runs[0]["rates"][metric], runs[1]["rates"][metric]
            delta = result["deltas"].get(metric)
            print(f"{metric:<14} {_fmt(a):>6} {_fmt(b):>6}   "
                  f"{'' if delta is None else f'{delta * 100:+.0f} pts'}")
        for flip in result["flips"]:
            print(f"  ↓ {flip['id']}: {flip['metric']} went from pass to fail")
        print(f"\nVerdict: {result['verdict']}")
        return

    _require_gateway()
    run = run_suite(args.prompt, use_judge=not args.no_judge,
                    progress=lambda i, n, r: print(f"  {i}/{n} {r['id']}", end="\r"))
    path = save_run(run)
    _print_run(run)
    print(f"\nSaved to {path.relative_to(config.BASE_DIR)} — compare with --compare v1 v2")


if __name__ == "__main__":
    main()
