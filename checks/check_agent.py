"""Self-check for the Agent exercises (A1 tools, A2 build, A3 run).

No LLM call needed for A1 checks — those are pure structure checks.
A2 and A3 need the LLM gateway (run `python -m checks.check_llm` first).

Run from the repo root:  python -m checks.check_agent
"""

from __future__ import annotations

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- A1: Tools are defined and decorated ----------------------------------
    print("-- A1: Tool definitions --")
    try:
        from src.agent import code_search, read_file, git_blame, REPO_ROOT

        # Check tools exist and are usable (callable or LangChain StructuredTool)
        for name, fn in [("code_search", code_search), ("read_file", read_file), ("git_blame", git_blame)]:
            is_tool = callable(fn) or hasattr(fn, "invoke")
            results.append(_check(f"{name} is callable", is_tool))

        # Check they are LangChain tools (have .name attribute from @tool)
        has_tool_attr = all(
            hasattr(fn, "name") or hasattr(fn, "tool_function")
            for fn in [code_search, read_file, git_blame]
        )
        if has_tool_attr:
            results.append(_check("tools are decorated with @tool (have .name)", True))
        else:
            # They might be plain functions (not yet decorated) — that's OK for
            # a partial check, but we note it
            print("   ⚠️  tools don't appear to be @tool-decorated yet — A1 is partially done")

        # Check read_file sandboxing (should reject paths outside repo)
        try:
            out = read_file.invoke({"path": "/etc/passwd"}) if hasattr(read_file, "invoke") else read_file("/etc/passwd")
            sandboxed = "error" in out.lower() or "outside" in out.lower()
            results.append(_check("read_file rejects paths outside repo (sandboxed)", sandboxed))
        except NotImplementedError:
            results.append(_check("read_file implemented", False))
            print("   ↳ still a TODO — see EXERCISES.md (A1b)")
        except Exception:
            results.append(_check("read_file rejects paths outside repo", True))

    except NotImplementedError:
        results.append(_check("tool functions implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (A1)")
    except ImportError as e:
        results.append(_check(f"imports work ({e})", False))
        print("   ↳ check that langchain dependencies are installed")

    # --- A2: build_agent returns a compiled graph with .invoke ----------------
    print("\n-- A2: build_agent --")
    try:
        from src.agent import build_agent

        agent = build_agent()
        has_invoke = hasattr(agent, "invoke")
        results.append(_check("build_agent returns an object with .invoke", has_invoke))
    except NotImplementedError:
        results.append(_check("build_agent implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (A2)")
    except Exception as e:
        results.append(_check(f"build_agent runs without error ({type(e).__name__}: {e})", False))

    # --- A3: run_agent produces a string answer -------------------------------
    print("\n-- A3: run_agent (needs LLM gateway) --")
    try:
        from src.agent import run_agent

        answer = run_agent("What files are in this repository?")
        got_answer = isinstance(answer, str) and len(answer) > 10
        results.append(_check("run_agent returns a non-empty string", got_answer))
    except NotImplementedError:
        results.append(_check("run_agent implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (A3)")
    except Exception as e:
        results.append(_check(f"run_agent runs without error ({type(e).__name__}: {e})", False))
        print("   ↳ if this is a gateway error, check: python -m checks.check_llm")

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
