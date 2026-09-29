"""Self-check for the Generation exercises (G1 core, G2/G3/G4 optional).

The generation layer calls a language model, so the *live* checks need the LiteLLM
gateway (verify it once with `python -m checks.check_llm`). But the important
safety behaviour is testable with NO key:

  - format_context numbering           (pure string)
  - G3 used_sources citation filtering (pure function)
  - G2 abstention                      (short-circuits BEFORE any model call)

These G exercises are SELF-CONTAINED: semantic_search (R1), confident_hits (R3)
and SessionMemory (M1) are provided by src/rag_helper.py, so you do NOT need the
retrieval/memory exercises done first. You only need the vector index built
(`python -m src.ingest`) so there's something to retrieve.

Run from the repo root:  python -m checks.check_rag
"""

from __future__ import annotations

import inspect
import sys

import chromadb

from src import config, rag
from src.rag import (
    ABSTAIN_MESSAGE,
    answer,
    answer_conversational,
    answer_or_abstain,
    format_context,
    used_sources,
)
# G4 takes any memory with add()/as_messages(); use the ready-made one from
# rag_helper so this check never depends on the M1 exercise being done.
from src.rag_helper import SessionMemory


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def _is_stub(fn) -> bool:
    """True if the function still raises NotImplementedError (i.e. not done)."""
    try:
        return "raise NotImplementedError" in inspect.getsource(fn)
    except OSError:
        return False


def _index_ready() -> bool:
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.COLLECTION_NAME).count() > 0
    except Exception:  # noqa: BLE001
        return False


def _gateway_configured() -> bool:
    # Mirrors checks/check_llm: a litellm_proxy model needs a proxy key.
    if config.LLM_MODEL.startswith("litellm_proxy/") and not config.LITELLM_PROXY_API_KEY:
        return False
    return True


# Three fake hits — enough to test the pure, key-free helpers deterministically.
FAKE_HITS = [
    {"text": "alpha", "source": "data/sample_company/a.md", "score": 0.9},
    {"text": "bravo", "source": "data/sample_company/b.md", "score": 0.8},
    {"text": "charlie", "source": "data/sample_company/c.md", "score": 0.7},
]


def _implementation_status() -> None:
    """Always-shown, key-free: which of G1–G4 are implemented vs still stubs."""
    print("-- Implementation status (G1–G4) --")
    for code, fn in [
        ("G1", rag.answer),
        ("G2", rag.answer_or_abstain),
        ("G3", rag.used_sources),
        ("G4", rag.answer_conversational),
    ]:
        done = not _is_stub(fn)
        state = "implemented" if done else "NOT implemented yet"
        print(f"{'✅' if done else '❌'} {code} {fn.__name__}() — {state}")


def main() -> None:
    results: list[bool] = []

    _implementation_status()

    # --- format_context (scaffolding, key-free) ------------------------------
    print("\n-- format_context (provided) --")
    ctx = format_context(FAKE_HITS)
    results.append(_check("numbers each hit and keeps its source ([1] … [2] … [3] …)",
                          "[1]" in ctx and "[3]" in ctx and "a.md" in ctx and "alpha" in ctx))

    # --- G3: used_sources (pure, key-free) -----------------------------------
    print("\n-- G3: used_sources (pure function, no key) --")
    try:
        two = used_sources("Use [1] and then [3] for details.", FAKE_HITS)
        results.append(_check("returns only the cited hits, in order ([1],[3] -> a.md, c.md)",
                              [h["source"] for h in two] == ["data/sample_company/a.md",
                                                             "data/sample_company/c.md"]))
        results.append(_check("no citations in the text -> empty list",
                              used_sources("No markers here.", FAKE_HITS) == []))
    except NotImplementedError:
        results.append(_check("used_sources implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (G3)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"used_sources runs without error ({type(e).__name__}: {e})", False))

    # Everything below leans on the vector index (retrieval).
    if not _index_ready():
        print("\n❌ No vector index found. Build it first:  python -m src.ingest")
        _summary(results)

    # --- G2: abstention (key-free — must NOT call the model) ------------------
    print("\n-- G2: answer_or_abstain — abstain path (no key) --")
    try:
        # min_score=0.99 forces "nothing is confident enough", so this must short-
        # circuit to the abstain message WITHOUT ever reaching the gateway.
        out = answer_or_abstain("How do I set up my local environment?", k=4, min_score=0.99)
        results.append(_check("abstains (returns ABSTAIN_MESSAGE) when nothing clears the bar",
                              out.get("answer") == ABSTAIN_MESSAGE))
        results.append(_check("abstention returns empty sources", out.get("sources") == []))
    except NotImplementedError:
        results.append(_check("answer_or_abstain implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (G2)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"answer_or_abstain runs without error ({type(e).__name__}: {e})", False))
        print("   ↳ tip: retrieval is provided by rag_helper — this is likely an index "
              "issue; rebuild with `python -m src.ingest`")

    passed_so_far = sum(bool(r) for r in results)
    total_required = len(results)

    # --- Live checks (need the gateway) --------------------------------------
    print("\n-- G1 / G2 / G4: live generation (needs the LiteLLM gateway) --")
    if not _gateway_configured():
        print("⚠️  Gateway not configured — skipping live checks. Copy .env.example to .env, "
              "then verify with `python -m checks.check_llm`.")
        print("   (Implementation status for G1/G2/G4 is shown at the top.)")
    else:
        # G1: grounded, cited answer to the setup question.
        try:
            res = answer("How do I set up my local environment?", k=4)
            ok_shape = isinstance(res, dict) and "answer" in res and "sources" in res
            print(f"{'✅' if ok_shape else '⚠️ '} G1 answer() returns {{answer, sources}}")
            if ok_shape:
                cites_setup = any("setup.md" in h["source"] for h in res["sources"])
                has_marker = "[1]" in res["answer"] or "[2]" in res["answer"]
                print(f"{'✅' if cites_setup else '⚠️ '} G1 retrieves setup.md as a source"
                      f"{'' if cites_setup else '  (unexpected — is the index up to date?)'}")
                print(f"{'✅' if has_marker else '⚠️ '} G1 answer contains a [n] citation marker"
                      f"{'' if has_marker else '  (nudge the SYSTEM_PROMPT / prompt if not)'}")
        except NotImplementedError:
            print("⚠️  G1 answer() not implemented yet — see EXERCISES.md (G1)")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  G1 errored ({type(e).__name__}: {e})")

        # G2 on-topic: should actually answer (not abstain) at the normal threshold.
        try:
            res = answer_or_abstain("How do I set up my local environment?", k=4, min_score=0.25)
            answered = res.get("answer") != ABSTAIN_MESSAGE and bool(res.get("sources"))
            print(f"{'✅' if answered else '⚠️ '} G2 answers on-topic questions (does not abstain)")
        except NotImplementedError:
            print("⚠️  G2 answer_or_abstain not implemented yet — see EXERCISES.md (G2)")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  G2 errored ({type(e).__name__}: {e})")

        # G4: conversational — memory grows by two turns and an answer comes back.
        try:
            mem = SessionMemory(window=6)
            res = answer_conversational("How do I set up my local environment?", mem, k=4)
            ok = bool(res.get("answer")) and len(mem.turns) == 2
            print(f"{'✅' if ok else '⚠️ '} G4 answers and records the turn (memory grew by 2)")
        except NotImplementedError:
            print("⚠️  G4 answer_conversational not implemented yet — see EXERCISES.md (G4)")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  G4 errored ({type(e).__name__}: {e})")

    _summary(results, passed_so_far, total_required)


def _summary(results: list[bool], passed: int | None = None, total: int | None = None) -> None:
    passed = sum(bool(r) for r in results) if passed is None else passed
    total = len(results) if total is None else total
    print(f"\n{passed}/{total} required (key-free) checks passed. "
          f"See 'Implementation status' at the top for G1–G4, and the live checks "
          f"above when the gateway is set up.")
    sys.exit(0 if results and passed == total else 1)


if __name__ == "__main__":
    main()
