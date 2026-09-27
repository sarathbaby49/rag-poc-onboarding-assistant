"""Self-check for the Retrieval exercises (R1 core, R2 stretch).

No API key needed — this is pure embeddings + math.

Run from the repo root:  python -m checks.check_retrieval
"""

from __future__ import annotations

import sys

import chromadb

from src import config
from src.retrieve import semantic_search, hybrid_search, confident_hits


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def _index_ready() -> bool:
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.COLLECTION_NAME).count() > 0
    except Exception:
        return False


def main() -> None:
    if not _index_ready():
        print("❌ No vector index found. Build it first:  python -m src.ingest")
        sys.exit(1)

    results: list[bool] = []

    # --- R1: semantic_search (required) --------------------------------------
    print("-- R1: semantic_search (required) --")
    try:
        hits = semantic_search("How do I set up my local environment?", k=4)
        results.append(_check("returns a list of k=4 hits", isinstance(hits, list) and len(hits) == 4))
        shape_ok = all(isinstance(h, dict) and {"text", "source", "score"} <= set(h) for h in hits)
        results.append(_check("each hit has text / source / score", shape_ok))
        if shape_ok:
            scores = [h["score"] for h in hits]
            results.append(_check("hits are sorted best-first (scores descending)",
                                  scores == sorted(scores, reverse=True)))
            results.append(_check("scores look like cosine similarity (0..1)",
                                  all(-0.01 <= s <= 1.01 for s in scores)))
            results.append(_check("the setup question retrieves setup.md",
                                  any("setup.md" in h["source"] for h in hits)))
    except NotImplementedError:
        results.append(_check("semantic_search implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (R1)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"semantic_search runs without error ({type(e).__name__}: {e})", False))

    # --- R2: hybrid_search (stretch, not counted) ----------------------------
    print("\n-- R2: hybrid_search (stretch, optional) --")
    try:
        hits = hybrid_search("issue_refresh_token", k=4)
        found = any("auth.py" in h["source"] for h in hits)
        print(f"{'✅' if found else '⚠️ '} 'issue_refresh_token' surfaces code/auth.py in top-k"
              f"{'' if found else '  (semantic alone often misses exact symbols — that is the point of hybrid)'}")
    except NotImplementedError:
        print("⚠️  not implemented yet (optional) — see EXERCISES.md (R2)")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  errored ({type(e).__name__}: {e})")

    # --- R3: confident_hits (extra, not counted) -----------------------------
    print("\n-- R3: confident_hits (extra, optional) --")
    try:
        q = "How do I set up my local environment?"
        loose = confident_hits(q, k=4, min_score=0.0)
        strict = confident_hits(q, k=4, min_score=0.99)
        ok = len(loose) == len(semantic_search(q, 4)) and len(strict) == 0
        print(f"{'✅' if ok else '⚠️ '} keeps hits at min_score=0.0, drops all at min_score=0.99"
              f"{'' if ok else '  (see EXERCISES.md R3)'}")
    except NotImplementedError:
        print("⚠️  not implemented yet (optional) — see EXERCISES.md (R3)")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  errored ({type(e).__name__}: {e})")

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} required checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
