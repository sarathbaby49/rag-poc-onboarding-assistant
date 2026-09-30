"""Self-check for the Memory exercises (M1 + M3 + M4).

No API key needed — this is pure Python (M4 uses the local embedding model).

Run from the repo root:  python -m checks.check_memory
"""

from __future__ import annotations

import sys

from src.memory import SessionMemory, SummaryBufferMemory, MemoryStore


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- M1: SessionMemory windowing -----------------------------------------
    print("-- M1: SessionMemory.as_messages --")
    try:
        m = SessionMemory(window=3)
        for i in range(5):
            m.add("user", f"msg{i}")
        msgs = m.as_messages()
        results.append(_check("returns only the last `window` (=3) turns", len(msgs) == 3))
        results.append(_check("keeps the MOST RECENT turns, in order",
                              [t["content"] for t in msgs] == ["msg2", "msg3", "msg4"]))
    except NotImplementedError:
        results.append(_check("as_messages implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (M1)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"as_messages runs without error ({type(e).__name__}: {e})", False))

    # --- M3: SummaryBufferMemory overflow -> summary -------------------------
    print("\n-- M3: SummaryBufferMemory.add --")
    try:
        sb = SummaryBufferMemory(window=2)
        for i in range(4):
            sb.add("user", f"msg{i}")
        results.append(_check("keeps only the last `window` (=2) turns verbatim", len(sb.turns) == 2))
        results.append(_check("keeps the MOST RECENT turns, in order",
                              [t["content"] for t in sb.turns] == ["msg2", "msg3"]))
        results.append(_check("folds overflow into the summary (oldest turns preserved)",
                              "msg0" in sb.summary and "msg1" in sb.summary))
        results.append(_check("as_context leads with the summary, then the live turns",
                              bool(sb.as_context()) and sb.as_context()[0]["role"] == "system"))
    except NotImplementedError:
        results.append(_check("add implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (M3)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"add runs without error ({type(e).__name__}: {e})", False))

    # --- M4: MemoryStore semantic recall -------------------------------------
    print("\n-- M4: MemoryStore.recall --")
    try:
        store = MemoryStore()
        store.remember("Sam is on the payments team.")            # relevant, added first
        store.remember("The office coffee machine is on floor 3.")
        store.remember("Standup is at 10am on weekdays.")         # most RECENT, but off-topic
        top = store.recall("which team am I part of?", k=1)
        results.append(_check("recall returns a list of texts", isinstance(top, list) and len(top) == 1))
        results.append(_check("recall ranks by MEANING, not recency (finds the payments memory)",
                              bool(top) and "payments" in top[0].lower()))
    except NotImplementedError:
        results.append(_check("recall implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (M4)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"recall runs without error ({type(e).__name__}: {e})", False))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} required checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
