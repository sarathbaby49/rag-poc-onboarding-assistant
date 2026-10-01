"""Self-check for the LangGraph exercises (G1 nodes, G2 routing, G3 graph).

Run from the repo root:  python -m checks.check_graph
"""

from __future__ import annotations

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- G1: Node functions ---------------------------------------------------
    print("-- G1: Node functions --")
    try:
        from src.graph import (
            plan, retrieve, check_grounding, replan, publish,
            PlanState, MAX_ATTEMPTS, _load_docs,
        )

        base: PlanState = {
            "user_request": "I'm a mid-level frontend dev, plan my first 2 weeks",
            "role": "",
            "days": 10,
            "plan": [],
            "gaps": [],
            "mentor_feedback": "",
            "approved": False,
            "attempts": 0,
            "messages": [],
        }

        # plan
        try:
            r = plan(base)
            results.append(_check("plan returns a dict", isinstance(r, dict)))
            results.append(_check("plan sets role", bool(r.get("role"))))
            results.append(_check("plan sets days > 0", r.get("days", 0) > 0))
            results.append(_check("plan creates a non-empty plan", len(r.get("plan", [])) > 0))
            results.append(_check("plan appends a welcome message", len(r.get("messages", [])) > 0))
        except NotImplementedError:
            results.append(_check("plan implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1a)")

        # retrieve (needs plan output)
        try:
            plan_out = plan(base)
            r = retrieve({**base, **plan_out})
            results.append(_check("retrieve returns a dict with 'plan'", "plan" in r))
            sources = [s.get("source") for s in r.get("plan", [])]
            results.append(_check(
                "retrieve finds at least some sources",
                any(s is not None for s in sources),
            ))
        except NotImplementedError:
            results.append(_check("retrieve implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1b)")

        # check_grounding
        try:
            plan_out = plan(base)
            ret_out = retrieve({**base, **plan_out})
            r = check_grounding({**base, **plan_out, **ret_out})
            results.append(_check("check_grounding returns gaps list", isinstance(r.get("gaps"), list)))
            results.append(_check(
                "check_grounding finds the 'compliance training' gap",
                any("compliance" in g.lower() or "training" in g.lower() for g in r.get("gaps", [])),
            ))
        except NotImplementedError:
            results.append(_check("check_grounding implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1c)")

        # replan
        try:
            plan_out = plan(base)
            ret_out = retrieve({**base, **plan_out})
            chk_out = check_grounding({**base, **plan_out, **ret_out})
            state_before_replan = {**base, **plan_out, **ret_out, **chk_out}
            r = replan(state_before_replan)
            results.append(_check("replan returns a dict", isinstance(r, dict)))
            results.append(_check("replan increments attempts", r.get("attempts", 0) == 1))
            results.append(_check("replan clears gaps", r.get("gaps") == []))
        except NotImplementedError:
            results.append(_check("replan implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1d)")

        # mentor_approval is tested structurally (interrupt is hard to test in isolation)
        try:
            from src.graph import mentor_approval
            # Just check it exists and is callable
            results.append(_check("mentor_approval exists and is callable", callable(mentor_approval) or hasattr(mentor_approval, "__call__")))
        except (NotImplementedError, ImportError):
            results.append(_check("mentor_approval implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1e)")

        # publish
        try:
            final_plan = [
                {"day": 1, "task": "Read README", "source": "README.md"},
                {"day": 2, "task": "Setup", "source": "setup.md"},
            ]
            pub_state = {**base, "plan": final_plan, "days": 2}
            r = publish(pub_state)
            results.append(_check("publish returns messages", len(r.get("messages", [])) > 0))
            results.append(_check(
                "publish includes day info",
                any("Day" in m or "day" in m for m in r.get("messages", [])),
            ))
        except NotImplementedError:
            results.append(_check("publish implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G1f)")

    except ImportError as e:
        results.append(_check(f"imports work ({e})", False))

    # --- G2: Routing ----------------------------------------------------------
    print("\n-- G2: Routing functions --")
    try:
        from src.graph import route_after_check, route_after_mentor, MAX_ATTEMPTS

        # route_after_check
        try:
            results.append(_check(
                "route_after_check → 'replan' when gaps + attempts < MAX",
                route_after_check({**base, "gaps": ["Day 4: Deploy"], "attempts": 0}) == "replan",
            ))
            results.append(_check(
                "route_after_check → 'mentor_approval' when no gaps",
                route_after_check({**base, "gaps": [], "attempts": 0}) == "mentor_approval",
            ))
            results.append(_check(
                "route_after_check → 'mentor_approval' when attempts >= MAX",
                route_after_check({**base, "gaps": ["x"], "attempts": MAX_ATTEMPTS}) == "mentor_approval",
            ))
        except NotImplementedError:
            results.append(_check("route_after_check implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G2a)")

        # route_after_mentor
        try:
            results.append(_check(
                "route_after_mentor → 'publish' when approved",
                route_after_mentor({**base, "approved": True}) == "publish",
            ))
            results.append(_check(
                "route_after_mentor → 'replan' when not approved",
                route_after_mentor({**base, "approved": False}) == "replan",
            ))
        except NotImplementedError:
            results.append(_check("route_after_mentor implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (G2b)")

    except ImportError as e:
        results.append(_check(f"routing imports ({e})", False))

    # --- G3: Build the graph --------------------------------------------------
    print("\n-- G3: build_graph --")
    try:
        from src.graph import build_graph

        graph = build_graph()
        results.append(_check("build_graph returns a compiled graph", hasattr(graph, "invoke")))

        # Check the graph has the expected nodes
        graph_obj = graph.get_graph()
        node_ids = set(graph_obj.nodes.keys())
        for expected in ("plan", "retrieve", "check_grounding", "replan", "mentor_approval", "publish"):
            results.append(_check(f"graph has '{expected}' node", expected in node_ids))

    except NotImplementedError:
        results.append(_check("build_graph implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (G3)")
    except Exception as e:
        results.append(_check(f"build_graph error ({type(e).__name__}: {e})", False))

    # --- Summary --------------------------------------------------------------
    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
