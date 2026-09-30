"""Layer 4 — MCP server (YOUR EXERCISES: MCP1 + MCP2).

MCP (Model Context Protocol) is how you expose your tools to an MCP client —
like Cursor, Claude Code, or any MCP-aware IDE — so a joinee can ask the
onboarding assistant questions *without leaving their editor*.

The idea: wrap the retrieval + code-search + git-blame tools you already have
as MCP tools. Then any MCP-aware client can discover and call them.

Concepts covered:
  - MCP server basics (FastMCP)
  - Exposing Python functions as MCP tools (@mcp.tool decorator)
  - MCP resources (read-only data endpoints)
  - Connecting to an IDE (Cursor / Claude Code config)

Reference: https://modelcontextprotocol.io  (Python SDK: `mcp`)

Self-check:  python -m checks.check_mcp
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src import config
from src.retrieve import semantic_search

# ── Repo root for sandboxing file reads ──────────────────────────────────────
REPO_ROOT = config.BASE_DIR


# --- MCP1: Create the MCP server and expose tools ----------------------------
# 1. Create a FastMCP server instance with a name
# 2. Expose search_docs, read_file, and git_blame as MCP tools
#
# Example:
#   mcp = FastMCP("onboarding-assistant")
#
#   @mcp.tool()
#   def search_docs(query: str) -> str:
#       """Search team docs and code for the given query."""
#       hits = semantic_search(query, k=4)
#       return "\n\n".join(f"{h['source']}:\n{h['text']}" for h in hits)
#
# Self-check:  python -m checks.check_mcp   (section MCP1)
# ─────────────────────────────────────────────────────────────────────────────

def create_mcp_server():
    """EXERCISE MCP1 — Create and return a FastMCP server with tools.

    Tools to expose:
      1. search_docs(query: str) -> str
         Semantic search over the onboarding docs and code.
         Use semantic_search(query, k=4) and format results.

      2. read_file(path: str) -> str
         Read a file from the repo (sandboxed to repo root).
         Reuse the read_file function from src/agent.py.

      3. git_blame(path: str, line: int) -> str
         Git blame a specific line.
         Reuse the git_blame function from src/agent.py.

    Return the FastMCP instance.

    Hint: Use @mcp.tool() decorator on each function. The docstring becomes
    the tool description that the IDE/model sees.
    """
    # TODO(MCP1): create the server and register tools
    raise NotImplementedError("Exercise MCP1: create the MCP server — see EXERCISES.md")


# --- MCP2: Add an MCP resource -----------------------------------------------
# Resources are read-only data endpoints. They're useful for things like
# "show me the onboarding checklist" without a tool call.
#
# Example:
#   @mcp.resource("onboarding://checklist")
#   def get_checklist() -> str:
#       return "1. Clone repo\n2. Run setup\n3. ..."
#
# Self-check:  python -m checks.check_mcp   (section MCP2)
# ─────────────────────────────────────────────────────────────────────────────

def add_resources(mcp):
    """EXERCISE MCP2 — Add resources to the MCP server.

    Add at least one resource:
      - onboarding://checklist — a newbie onboarding checklist
      - onboarding://architecture — a brief architecture overview

    Hint: use @mcp.resource("onboarding://checklist") decorator.
    """
    # TODO(MCP2): add resources to the server
    raise NotImplementedError("Exercise MCP2: add MCP resources — see EXERCISES.md")


# --- Entry point (run the server) ---------------------------------------------
# After implementing MCP1, this is how you run it:
#
#   python -m src.mcp_server          # starts the MCP server on stdio
#
# To connect from Cursor, add to .cursor/mcp.json:
#   {
#     "mcpServers": {
#       "onboarding-assistant": {
#         "command": "python",
#         "args": ["-m", "src.mcp_server"],
#         "cwd": "<repo-root>"
#       }
#     }
#   }

if __name__ == "__main__":
    server = create_mcp_server()
    # MCP2 is optional — if you've done it, call add_resources(server) here
    # add_resources(server)
    server.run()
