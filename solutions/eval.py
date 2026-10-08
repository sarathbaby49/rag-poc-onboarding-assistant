"""REFERENCE SOLUTION for src/eval/run_eval.py — Lab 1, Exercises EV1–EV4.

Try each yourself first! If you're stuck or out of time, copy the relevant
function body into src/eval/run_eval.py.

  - EV1 -> grade_keywords      (code-based grader)
  - EV2 -> retrieval_hit       (retrieval scored on its own)
  - EV3 -> judge_faithfulness  (LLM-as-judge)
  - EV4 -> compare_runs        (regression gate: SHIP / BLOCK)
"""

from __future__ import annotations

import json
import re

from src import config
from src.eval.run_eval import JUDGE_PROMPT, PASS_COVERAGE, is_abstention
from src.llm import complete
from src.rag import format_context
from src.rag_helper import semantic_search


# --- EV1 ------------------------------------------------------------------------
def grade_keywords(response: dict, case: dict) -> dict:
    if case.get("expect_abstain"):
        leaked = any(s.lower() in response["answer"].lower() for s in case.get("must_not_contain", []))
        passed = is_abstention(response["answer"]) and not leaked
        return {"coverage": 1.0 if passed else 0.0, "cited": True, "passed": passed}

    text = response["answer"].lower()
    points = case["expected_points"]
    coverage = sum(p.lower() in text for p in points) / len(points)
    cited = any(case["expected_source"] in h["source"] for h in response["sources"])
    return {"coverage": coverage, "cited": cited, "passed": coverage >= PASS_COVERAGE and cited}


# --- EV2 ------------------------------------------------------------------------
def retrieval_hit(case: dict, k: int = config.TOP_K) -> bool | None:
    if not case.get("expected_source"):
        return None
    hits = semantic_search(case["question"], k)
    return any(case["expected_source"] in h["source"] for h in hits)


# --- EV3 ------------------------------------------------------------------------
def judge_faithfulness(question: str, response: dict) -> dict:
    if response.get("abstained"):
        return {"verdict": "PASS", "reasoning": "abstained without calling the model"}

    prompt = JUDGE_PROMPT.format(
        context=format_context(response["sources"]), question=question, answer=response["answer"]
    )
    text = complete([{"role": "user", "content": prompt}], model=config.JUDGE_MODEL,
                    max_tokens=400, timeout=config.LLM_TIMEOUT)
    blocks = re.findall(r"\{.*?\}", text, re.DOTALL)
    try:
        data = json.loads(blocks[-1])
        return {"verdict": str(data["verdict"]).upper(), "reasoning": data.get("reasoning", "")}
    except (IndexError, KeyError, json.JSONDecodeError):
        return {"verdict": "FAIL", "reasoning": "judge returned no valid JSON"}


# --- EV4 ------------------------------------------------------------------------
def compare_runs(baseline: dict, candidate: dict, tolerance: float = 0.0) -> dict:
    deltas = {
        metric: candidate["rates"][metric] - rate
        for metric, rate in baseline["rates"].items()
        if rate is not None and candidate["rates"].get(metric) is not None
    }
    regressions = [m for m, d in deltas.items() if d < -tolerance]

    candidate_cases = {c["id"]: c for c in candidate["cases"]}
    flips = []
    for case in baseline["cases"]:
        other = candidate_cases.get(case["id"])
        if not other:
            continue
        for metric, passed in case["checks"].items():
            if passed is True and other["checks"].get(metric) is False:
                flips.append({"id": case["id"], "metric": metric})

    return {
        "deltas": deltas,
        "regressions": regressions,
        "flips": flips,
        "verdict": "BLOCK" if regressions or flips else "SHIP",
    }
