"""Self-check for the MCP server exercises (MCP1 server + tools, MCP2 resources).

No LLM call needed — this checks that the server object is created and has
the right tools registered.

Run from the repo root:  python -m checks.check_mcp
"""

from __future__ import annotations

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main() -> None:
    results: list[bool] = []

    # --- MCP1: Server creation and tools --------------------------------------
    print("-- MCP1: MCP server with tools --")
    try:
        from src.mcp_server import create_mcp_server

        server = create_mcp_server()
        results.append(_check("create_mcp_server returns an object", server is not None))

        # Check the server has the right type (FastMCP)
        type_name = type(server).__name__
        results.append(_check(
            f"server is a FastMCP instance (got {type_name})",
            "FastMCP" in type_name or "MCP" in type_name,
        ))

        # Check tools are registered — FastMCP stores them internally
        # We check by looking at the server's attributes
        has_tools = False
        tool_names: list[str] = []
        if hasattr(server, "_tool_manager"):
            tools = server._tool_manager._tools if hasattr(server._tool_manager, "_tools") else {}
            tool_names = list(tools.keys())
            has_tools = len(tool_names) >= 3
        elif hasattr(server, "_tools"):
            tool_names = list(server._tools.keys()) if isinstance(server._tools, dict) else []
            has_tools = len(tool_names) >= 3

        if tool_names:
            results.append(_check(
                f"server has >= 3 tools registered ({', '.join(tool_names)})",
                has_tools,
            ))
            results.append(_check(
                "search_docs tool is registered",
                "search_docs" in tool_names,
            ))
            results.append(_check(
                "read_file tool is registered",
                "read_file" in tool_names,
            ))
            results.append(_check(
                "git_blame tool is registered",
                "git_blame" in tool_names,
            ))
        else:
            print("   ⚠️  Could not inspect registered tools (FastMCP internals may have changed).")
            print("   Manual check: run `python -m src.mcp_server` and connect from Cursor.")

    except NotImplementedError:
        results.append(_check("create_mcp_server implemented", False))
        print("   ↳ still a TODO — see EXERCISES.md (MCP1)")
    except ImportError as e:
        results.append(_check(f"imports work ({e})", False))
        print("   ↳ run: pip install mcp")
    except Exception as e:
        results.append(_check(f"create_mcp_server runs without error ({type(e).__name__}: {e})", False))

    # --- MCP2: Resources (optional) -------------------------------------------
    print("\n-- MCP2: MCP resources (optional) --")
    try:
        from src.mcp_server import add_resources, create_mcp_server
        server = create_mcp_server()
        add_resources(server)
        print("✅ add_resources ran without error")

        # Try to inspect registered resources
        has_resources = False
        if hasattr(server, "_resource_manager"):
            resources = server._resource_manager._resources if hasattr(server._resource_manager, "_resources") else {}
            has_resources = len(resources) > 0
            if has_resources:
                print(f"   Found resources: {', '.join(resources.keys())}")
        elif hasattr(server, "_resources"):
            has_resources = len(server._resources) > 0

        if has_resources:
            print("✅ Resources are registered")
        else:
            print("   ⚠️  Could not verify resources (FastMCP internals may vary).")

    except NotImplementedError:
        print("⚠️  not implemented yet (optional) — see EXERCISES.md (MCP2)")
    except Exception as e:
        print(f"⚠️  errored ({type(e).__name__}: {e})")

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if results and passed == len(results) else 1)


if __name__ == "__main__":
    main()
