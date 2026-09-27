"""Layer 4 — MCP server (STUB / your mission).

MCP (Model Context Protocol) is how you expose your tools to an MCP client —
like Claude Code or an IDE — so a joinee can ask the onboarding assistant
questions *without leaving their editor*.

The idea: wrap the retrieval + code-search tools you already have as MCP tools.
Then any MCP-aware client can call them.

Your mission:
  1. `pip install mcp`.
  2. Define an MCP server that exposes `search_docs` and `read_file` tools,
     backed by src.retrieve / src.agent.
  3. Register it in your IDE / Claude Code and query it live.

Reference: https://modelcontextprotocol.io  (Python SDK: `mcp`)
"""

# from mcp.server.fastmcp import FastMCP
# from src.retrieve import semantic_search
#
# mcp = FastMCP("onboarding-assistant")
#
# @mcp.tool()
# def search_docs(query: str) -> str:
#     """Search team docs and code for the given query."""
#     hits = semantic_search(query, k=4)
#     return "\n\n".join(f"{h['source']}:\n{h['text']}" for h in hits)
#
# if __name__ == "__main__":
#     mcp.run()

raise NotImplementedError("Your mission: expose the tools as an MCP server.")
