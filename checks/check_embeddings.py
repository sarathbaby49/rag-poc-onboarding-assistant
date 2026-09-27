"""Self-check for the Embeddings exercise (E1).

No API key needed. First run downloads the embedding model (~90 MB) if you
haven't already run ingest.

Run from the repo root:  python -m checks.check_embeddings
"""

from __future__ import annotations

import sys

from src.embeddings import embed, cosine_similarity


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    print("-- E1: cosine_similarity --")
    try:
        # Simple, model-free vectors first (exact expected values).
        results.append(_check("identical vectors -> 1.0",
                              abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6))
        results.append(_check("perpendicular vectors -> 0.0",
                              abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6))
        results.append(_check("un-normalized vectors handled ([1,1] vs [2,2] -> 1.0)",
                              abs(cosine_similarity([1.0, 1.0], [2.0, 2.0]) - 1.0) < 1e-6))

        # Real embeddings: related pair should beat an unrelated pair.
        q = embed("how do I log in?")
        related = cosine_similarity(q, embed("authenticate a user"))
        unrelated = cosine_similarity(q, embed("refund a payment"))
        print(f"   related={related:.3f}  unrelated={unrelated:.3f}")
        results.append(_check("related meaning scores higher than unrelated", related > unrelated))
    except NotImplementedError:
        results.append(_check("cosine_similarity implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (E1)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"cosine_similarity runs without error ({type(e).__name__}: {e})", False))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} required checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
