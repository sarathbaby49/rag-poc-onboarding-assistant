"""Layer 4 — Setup-troubleshooting agent with tools (YOUR EXERCISES: A1 + A2 + A3).

RAG answers questions. An *agent* takes actions. For troubleshooting a broken
setup, the assistant needs to look things up dynamically:

    - code_search(query)      -> find where something is defined/used
    - read_file(path)         -> read a specific file
    - git_blame(path, line)   -> who last touched this line, and why

We use **LangChain** tool-calling with the LiteLLM gateway so the agent works
with any model behind the proxy — no direct provider key needed.

The pattern:
  1. Define tools as Python functions decorated with @tool.
  2. Pass the tools + a ChatOpenAI model to create_agent().
  3. Run the agent loop: the model decides which tools to call, you execute them,
     feed results back, loop until it has a final answer.

Concepts covered:
  - LangChain tools (@tool decorator, StructuredTool)
  - LangChain chat models (ChatOpenAI)
  - Agent graph (create_agent — the modern LangChain agent builder)

Self-check:  python -m checks.check_agent
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src import config
from src.retrieve import semantic_search

# ── Repo root for sandboxing file reads ──────────────────────────────────────
REPO_ROOT = config.BASE_DIR


# --- A1: Define tools --------------------------------------------------------
# Use the @tool decorator to create three tools:
#   1. code_search(query: str) -> str  — semantic search over docs & code
#   2. read_file(path: str) -> str     — read a file (sandboxed to repo)
#   3. git_blame(path: str, line: int) -> str — git blame a specific line
#
# Example:
#   @tool
#   def code_search(query: str) -> str:
#       """Search the codebase and docs for a keyword or concept."""
#       ...
#
# Self-check:  python -m checks.check_agent   (section A1)
# ─────────────────────────────────────────────────────────────────────────────

def code_search(query: str) -> str:
    """EXERCISE A1a — Search the codebase and docs for a keyword or concept.

    Use semantic_search(query, k=3) and format the results as readable text.
    Then decorate this function with @tool so LangChain can use it.

    Hint: semantic_search returns [{"text", "source", "score"}, ...].
    """
    # TODO(A1a): implement and decorate with @tool
    raise NotImplementedError("Exercise A1a: implement code_search tool — see EXERCISES.md")


def read_file(path: str) -> str:
    """EXERCISE A1b — Read the full contents of a file by its repo-relative path.

    IMPORTANT: sandbox this — only allow reading files inside REPO_ROOT.
    Return the file contents, or an error message if the file doesn't exist
    or is outside the repo.

    Hint: resolve the path, check it starts with REPO_ROOT, then open & read.
    """
    # TODO(A1b): implement and decorate with @tool
    raise NotImplementedError("Exercise A1b: implement read_file tool — see EXERCISES.md")


def git_blame(path: str, line: int) -> str:
    """EXERCISE A1c — Run git blame on a specific line of a file.

    Use subprocess to run `git blame -L {line},{line} -- {path}` and return
    the output. Sandbox the path to REPO_ROOT.

    Hint: subprocess.run([...], capture_output=True, text=True, cwd=REPO_ROOT)
    """
    # TODO(A1c): implement and decorate with @tool
    raise NotImplementedError("Exercise A1c: implement git_blame tool — see EXERCISES.md")


# --- A2: Build the agent -----------------------------------------------------
# Create a function that:
#   1. Initializes a ChatOpenAI model (using config.LLM_MODEL)
#   2. Passes the model, tools, and system prompt to create_agent()
#   3. Returns the compiled agent graph
#
# Self-check:  python -m checks.check_agent   (section A2)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Setup Troubleshooting Agent for an engineering team.
You help new joinees debug environment issues, find code, and understand the repo.

You have tools to search code/docs, read files, and check git blame.
Use them step by step to investigate and answer the question.
Be concise and give actionable advice."""


def build_agent():
    """EXERCISE A2 — Create an agent graph with the three tools.

    Steps:
      1. Create a ChatOpenAI model with the proxy settings from config.
      2. Call create_agent() with the model, tools list, and system_prompt.
      3. Return the compiled agent graph.

    Self-check:  python -m checks.check_agent   (section A2)
    """
    # TODO(A2): build and return an agent graph
    raise NotImplementedError("Exercise A2: build the agent — see EXERCISES.md")


# --- A3: Run the agent -------------------------------------------------------
def run_agent(question: str) -> str:
    """EXERCISE A3 — Invoke the agent on a question and return the answer.

    Steps:
      1. Call build_agent() to get the agent graph.
      2. Invoke it with {"messages": [{"role": "user", "content": question}]}.
      3. Return the last message's content from the result.

    Self-check:  python -m checks.check_agent   (section A3)
    """
    # TODO(A3): invoke the agent and return the answer
    raise NotImplementedError("Exercise A3: run the agent — see EXERCISES.md")


# --- CLI entry point ----------------------------------------------------------
def _cli() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or "Where is the payment provider code and who last changed it?"
    print(f"Q: {question}\n")
    print(run_agent(question))


if __name__ == "__main__":
    _cli()
