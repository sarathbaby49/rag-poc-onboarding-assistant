"""Layer 6 — Evaluation (STUB / your mission).

"It worked when I tried it" is not evaluation. A golden set is a fixed list of
real questions + what a good answer must contain. You run the assistant against
it every time you change something, and watch the score.

golden_set.jsonl has the questions. This runner shows the shape; the grading and
the end-to-end "did setup actually work" test are yours to implement.

Run:  python -m src.eval.run_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag import answer

GOLDEN = Path(__file__).parent / "golden_set.jsonl"


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def grade(response: dict, case: dict) -> float:
    """TODO: score one answer 0..1.

    Two common approaches:
      A. Keyword/point coverage — did the answer contain expected_points, and did
         it cite expected_source?  (cheap, deterministic, good enough to start)
      B. LLM-as-judge — ask a model to rate faithfulness/completeness 0..1.
         (better for nuance; costs a call per case)
    """
    text = response["answer"].lower()
    got_points = sum(p.lower() in text for p in case["expected_points"])
    cited = any(case["expected_source"] in h["source"] for h in response["sources"])
    # naive placeholder score — replace with a real rubric
    return (got_points / len(case["expected_points"])) * (1.0 if cited else 0.5)


def main() -> None:
    cases = load_golden()
    scores = []
    for case in cases:
        resp = answer(case["question"])
        s = grade(resp, case)
        scores.append(s)
        print(f"[{s:.2f}] {case['question']}")
    print(f"\nAverage score: {sum(scores) / len(scores):.2f} over {len(scores)} cases")
    # TODO: add an end-to-end "setup actually works" test that runs the commands
    # from setup.md in a clean container and checks the app starts.


if __name__ == "__main__":
    main()
