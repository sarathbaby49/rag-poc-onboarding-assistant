"""REFERENCE SOLUTION for src/agent.py (Exercises A1 + A2 + A3).

Setup-troubleshooting agent with LangChain tools, backed by the LiteLLM gateway.

Try the exercise first! If you're stuck or out of time, copy the relevant
function into src/agent.py.
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

REPO_ROOT = config.BASE_DIR

SYSTEM_PROMPT = """You are a Setup Troubleshooting Agent for an engineering team.
You help new joinees debug environment issues, find code, and understand the repo.

You have tools to search code/docs, read files, and check git blame.
Use them step by step to investigate and answer the question.
Be concise and give actionable advice."""


# --- A1: Tools ---------------------------------------------------------------

@tool
def code_search(query: str) -> str:
    """Search the codebase and docs for a keyword or concept."""
    hits = semantic_search(query, k=3)
    if not hits:
        return "No results found."
    return "\n\n".join(
        f"📄 {h['source']} (score: {h['score']:.2f}):\n{h['text']}" for h in hits
    )


@tool
def read_file(path: str) -> str:
    """Read the full contents of a file by its repo-relative path."""
    resolved = (REPO_ROOT / path).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        return "Error: path is outside the repository."
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: {e}"


@tool
def git_blame(path: str, line: int) -> str:
    """Run git blame on a specific line of a file to see who last changed it."""
    resolved = (REPO_ROOT / path).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        return "Error: path is outside the repository."
    try:
        result = subprocess.run(
            ["git", "blame", "-L", f"{line},{line}", "--", str(resolved)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=10,
        )
        return result.stdout.strip() or result.stderr.strip() or "No blame output."
    except subprocess.TimeoutExpired:
        return "Error: git blame timed out."
    except OSError as e:
        return f"Error: {e}"

TOOLS = [code_search, read_file, git_blame]


# --- A2: Build the agent -----------------------------------------------------

def build_agent():
    """Create an agent graph with the three tools using create_agent."""
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LITELLM_PROXY_API_BASE or None,
        api_key=config.LITELLM_PROXY_API_KEY or None,
        max_tokens=config.MAX_TOKENS,
    )

    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="setup-agent",
    )


# --- A3: Run the agent -------------------------------------------------------

def run_agent(question: str) -> str:
    """Invoke the agent on a question and return the answer."""
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    return result["messages"][-1].content


# --- CLI ----------------------------------------------------------------------
def _cli() -> None:
    import sys

    question = " ".join(sys.argv[1:]) or "Where is the payment provider code and who last changed it?"
    print(f"Q: {question}\n")
    answer = run_agent(question)
    print(f"\nA: {answer}")


if __name__ == "__main__":
    _cli()
