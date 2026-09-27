"""Regression check for the default chunker (boundary-aware `chunk_text_smart`).

Chunking ships implemented, so this just guards that the default chunker keeps
words whole. No API key needed. Run from the repo root:  python -m checks.check_ingest
"""

from __future__ import annotations

import sys

from src.ingest import chunk_text_smart


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    print("-- chunk_text_smart (boundary-aware, default) --")
    # 300 whole words; fixed-size slicing would cut words, boundary-aware must not.
    text = " ".join(["word"] * 300)
    try:
        chunks = chunk_text_smart(text, size=100, overlap=20)
        results.append(_check("returns more than one chunk", isinstance(chunks, list) and len(chunks) > 1))
        no_cut = all(all(tok == "word" for tok in c.split()) for c in chunks)
        results.append(_check("never splits a word mid-token", no_cut))
        results.append(_check("each chunk respects ~size (<= size)", all(len(c) <= 100 for c in chunks)))
        short = chunk_text_smart("just a short line", size=100, overlap=20)
        results.append(_check("text shorter than size -> single chunk", short == ["just a short line"]))
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"chunk_text_smart runs without error ({type(e).__name__}: {e})", False))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} required checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
