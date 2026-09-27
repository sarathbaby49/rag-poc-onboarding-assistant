"""Self-check for the Memory exercises (M1 + M2).

No API key needed — this is pure Python.

Run from the repo root:  python -m checks.check_memory
"""

from __future__ import annotations

import sys

from src.memory import SessionMemory, JoineeProfile


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

    # --- M2: JoineeProfile persistence ---------------------------------------
    print("\n-- M2: JoineeProfile.save / load --")
    try:
        p = JoineeProfile(name="_selfcheck_user", role="engineer")
        p.completed_steps = ["setup", "first-pr"]
        p.save()
        reloaded = JoineeProfile.load("_selfcheck_user")
        results.append(_check("profile round-trips through disk (save + load)",
                              reloaded.completed_steps == ["setup", "first-pr"]))
        results.append(_check("name and role persisted",
                              reloaded.name == "_selfcheck_user" and reloaded.role == "engineer"))
    except NotImplementedError:
        results.append(_check("save/load implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (M2)")
    except Exception as e:  # noqa: BLE001
        results.append(_check(f"save/load run without error ({type(e).__name__}: {e})", False))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} required checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
