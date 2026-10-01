"""REFERENCE SOLUTION for src/mcp_server.py (Exercises MCP1 + MCP2).

MCP server exposing the onboarding tools for use inside an IDE.
Try the exercise first! If you're stuck or out of time, copy the relevant
parts into src/mcp_server.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src import config
from src.retrieve import semantic_search

REPO_ROOT = config.DATA_DIR


# --- MCP1: Create the server and tools ----------------------------------------

mcp = FastMCP("onboarding-assistant")


@mcp.tool()
def search_docs(query: str) -> str:
    """Search team docs and code for the given query.

    Uses semantic search over the onboarding corpus (docs, code, Jira, Slack).
    Returns the top matches with source and relevance score.
    """
    hits = semantic_search(query, k=4)
    if not hits:
        return "No relevant results found."
    return "\n\n".join(
        f"📄 {h['source']} (score: {h['score']:.2f}):\n{h['text']}" for h in hits
    )


@mcp.tool()
def read_file(path: str) -> str:
    """Read the full contents of a file by its repo-relative path.

    The path is sandboxed to the repository root for security.
    """
    resolved = (REPO_ROOT / path).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        return "Error: path is outside the repository."
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: {e}"


@mcp.tool()
def git_blame(path: str, line: int) -> str:
    """Run git blame on a specific line of a file.

    Shows who last modified the line and when, useful for finding the right
    person to ask about a piece of code.
    """
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


# --- MCP2: Resources ----------------------------------------------------------

@mcp.resource("onboarding://checklist")
def get_checklist() -> str:
    """New engineer onboarding checklist."""
    return """# 🧭 Onboarding Checklist

## Day 1 — Environment Setup
- [ ] Clone the repository
- [ ] Create and activate virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Build the vector index (`python -m src.ingest`)
- [ ] Run the retrieval lab (`streamlit run retrieval_lab.py`)
- [ ] Verify the LLM gateway (`python -m checks.check_llm`)

## Day 2 — Architecture Tour
- [ ] Read the payment service docs
- [ ] Read the auth service docs
- [ ] Explore the codebase with `explore.py`
- [ ] Understand the data flow: ingest → embed → store → retrieve → generate

## Day 3 — First Ticket
- [ ] Pick a ticket from the Jira board
- [ ] Create a feature branch
- [ ] Write + test the change
- [ ] Open your first PR
"""


@mcp.resource("onboarding://architecture")
def get_architecture() -> str:
    """Brief architecture overview of Acme Shop."""
    return """# 🏗️ Acme Shop Architecture

## Services
- **Payment Service** (`code/payment_providers.py`): Razorpay + Stripe
- **Auth Service** (`code/auth.py`): JWT authentication with refresh tokens
- **Data Layer**: PostgreSQL (transactions), Redis (sessions)

## RAG Pipeline
1. **Ingest** (`src/ingest.py`): chunk docs → embed → store in ChromaDB
2. **Retrieve** (`src/retrieve.py`): query → embed → nearest neighbors
3. **Generate** (`src/rag.py`): context + question → grounded answer

## Key Config
- Embedding model: `all-MiniLM-L6-v2` (384-dim, local)
- Vector store: ChromaDB (on-disk at `.chroma/`)
- LLM: via LiteLLM proxy (model configurable in `.env`)
"""


def create_mcp_server() -> FastMCP:
    """Return the configured MCP server (for use in src/mcp_server.py)."""
    return mcp


if __name__ == "__main__":
    mcp.run()
