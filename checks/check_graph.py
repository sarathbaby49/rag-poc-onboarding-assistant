"""Self-check for the LangGraph exercises (G1 nodes, G2 routing, G3 graph).

No LLM call needed — the graph nodes use canned responses in the reference
solution, so this checks structure and flow.

Run from the repo root:  python -m checks.check_graph
"""

from __future__ import annotations

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- G1: Node functions exist and return dicts ----------------------------
    print("-- G1: Node functions --")
    try:
        from src.graph import welcome, setup_help, mentor_checkpoint, architecture_tour, OnboardingState

        base_state: OnboardingState = {
            "joinee": "TestUser",
            "role": "backend",
            "stage": "welcome",
            "messages": [],
            "needs_mentor": False,
            "mentor_notes": "",
            "completed_steps": [],
        }

        # welcome
        try:
            w = welcome(base_state)
            results.append(_check("welcome returns a dict", isinstance(w, dict)))
            results.append(_check(
                "welcome sets stage to 'setup'",
                w.get("stage") == "setup",
            ))
            results.append(_check(
                "welcome adds to completed_steps",
                "welcome" in w.get("completed_steps", []),
            ))
            results.append(_check(
                "welcome adds a message",
                len(w.get("messages", [])) > len(base_state["messages"]),
            ))
        except NotImplementedError:
            results.append(_check("welcome implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1a)")

        # setup_help (happy path — no stuck keywords)
        try:
            setup_state = {**base_state, "stage": "setup", "messages": [
                {"role": "user", "content": "How do I set up my local env?"}
            ]}
            s = setup_help(setup_state)
            results.append(_check("setup_help returns a dict", isinstance(s, dict)))
            results.append(_check(
                "setup_help adds a message",
                len(s.get("messages", [])) > len(setup_state["messages"]),
            ))
            results.append(_check(
                "setup_help does NOT flag mentor for normal question",
                s.get("needs_mentor") is False or s.get("needs_mentor") is None,
            ))
        except NotImplementedError:
            results.append(_check("setup_help implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1b)")

        # setup_help (stuck path)
        try:
            stuck_state = {**base_state, "stage": "setup", "messages": [
                {"role": "user", "content": "I'm getting an error when I run pip install, it's broken!"}
            ]}
            s2 = setup_help(stuck_state)
            results.append(_check(
                "setup_help flags needs_mentor when user is stuck",
                s2.get("needs_mentor") is True,
            ))
        except NotImplementedError:
            pass  # already counted above

        # mentor_checkpoint
        try:
            ck_state = {**base_state, "stage": "setup", "needs_mentor": True, "mentor_notes": "Looks good!"}
            c = mentor_checkpoint(ck_state)
            results.append(_check("mentor_checkpoint returns a dict", isinstance(c, dict)))
            results.append(_check(
                "mentor_checkpoint clears needs_mentor",
                c.get("needs_mentor") is False,
            ))
            results.append(_check(
                "mentor_checkpoint advances stage to 'architecture'",
                c.get("stage") == "architecture",
            ))
        except NotImplementedError:
            results.append(_check("mentor_checkpoint implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1c)")

        # architecture_tour
        try:
            arch_state = {**base_state, "stage": "architecture"}
            a = architecture_tour(arch_state)
            results.append(_check("architecture_tour returns a dict", isinstance(a, dict)))
            results.append(_check(
                "architecture_tour sets stage to 'first_ticket'",
                a.get("stage") == "first_ticket",
            ))
        except NotImplementedError:
            results.append(_check("architecture_tour implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1d)")

    except ImportError as e:
        results.append(_check(f"imports work ({e})", False))

    # --- G2: Routing function -------------------------------------------------
    print("\n-- G2: route_after_setup --")
    try:
        from src.graph import route_after_setup

        mentor_state = {**base_state, "needs_mentor": True}
        ok_state = {**base_state, "needs_mentor": False}

        results.append(_check(
            "routes to 'checkpoint' when needs_mentor=True",
            route_after_setup(mentor_state) == "checkpoint",
        ))
        results.append(_check(
            "routes to 'architecture' when needs_mentor=False",
            route_after_setup(ok_state) == "architecture",
        ))
    except NotImplementedError:
        results.append(_check("route_after_setup implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (G2)")
    except Exception as e:
        results.append(_check(f"route_after_setup runs without error ({type(e).__name__}: {e})", False))

    # --- G3: Build and run the graph ------------------------------------------
    print("\n-- G3: build_graph --")
    try:
        from src.graph import build_graph

        graph = build_graph()
        results.append(_check("build_graph returns a compiled graph", hasattr(graph, "invoke")))

        # Run the happy path (no mentor needed)
        happy_state: OnboardingState = {
            "joinee": "TestUser",
            "role": "backend",
            "stage": "welcome",
            "messages": [{"role": "user", "content": "How do I get started?"}],
            "needs_mentor": False,
            "mentor_notes": "",
            "completed_steps": [],
        }
        result = graph.invoke(happy_state)
        results.append(_check(
            "happy path reaches 'first_ticket' stage",
            result.get("stage") == "first_ticket",
        ))
        results.append(_check(
            "happy path completes welcome + architecture",
            "welcome" in result.get("completed_steps", [])
            and "architecture" in result.get("completed_steps", []),
        ))

    except NotImplementedError:
        results.append(_check("build_graph implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (G3)")
    except Exception as e:
        results.append(_check(f"build_graph runs without error ({type(e).__name__}: {e})", False))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
