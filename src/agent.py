"""Layer 4 — Agent with tools (STUB / your mission).

RAG answers questions. An *agent* takes actions. For troubleshooting a broken
setup, the assistant needs to look things up dynamically:

    - code_search(query)      -> find where something is defined/used
    - read_file(path)         -> read a specific file
    - git_blame(path, line)   -> who last touched this line, and why

The pattern (Claude tool use): you describe tools, Claude decides which to call,
you execute them, feed results back, loop until it has an answer.

This stub defines the tool schemas and handlers. Wire them into a tool-use loop
(see the Anthropic SDK "tool use" docs, or use client.beta.messages.tool_runner)
to complete Layer 4.
"""

from __future__ import annotations

import subprocess

from src.retrieve import semantic_search

# --- Tool schemas (what Claude sees) ----------------------------------------
TOOLS = [
    {
        "name": "code_search",
        "description": "Search the codebase and docs for a keyword or concept.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full contents of a file by its repo-relative path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    # TODO: add a git_blame tool schema.
]


# --- Tool handlers (what actually runs) --------------------------------------
def code_search(query: str) -> str:
    hits = semantic_search(query, k=3)
    return "\n\n".join(f"{h['source']}:\n{h['text']}" for h in hits)


def read_file(path: str) -> str:
    # TODO: sandbox this — only allow reading inside the repo (Layer 6 security).
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"


def git_blame(path: str, line: int) -> str:
    # TODO: run `git blame` safely and return the relevant author/commit.
    raise NotImplementedError("Your mission: implement git_blame as a tool.")


def run_agent(question: str) -> str:
    """TODO: implement the tool-use loop.

    Skeleton of the loop:
      1. Send question + TOOLS to Claude.
      2. While response.stop_reason == "tool_use": run each requested tool,
         append tool_result blocks, call again.
      3. Return the final text answer.
    """
    raise NotImplementedError("Your mission: build the agent tool-use loop.")
