# 🤖 Agent Harness Demo — Claude Agent SDK via LiteLLM

A single notebook, [`claude_agent_harness.ipynb`](claude_agent_harness.ipynb), that
shows what an **agent harness** is: everything around the model that turns it into
a controlled agent. The use case is a *Sales Analyst Agent* that explores a CSV,
calls business tools, writes a report and gets reviewed.

| Section | Harness concept |
|---|---|
| 1 | The agent loop with built-in tools (`Read`, `Write`, `Bash`, `Glob`) |
| 2 | Custom tools as an in-process MCP server (`@tool`) |
| 3 | Hooks: a guardrail that blocks dangerous commands, plus an audit log |
| 4 | Permissions: allowed tools, permission mode, working-directory sandbox |
| 5 | Multi-turn sessions with `ClaudeSDKClient` (the agent remembers context) |
| 6 | Subagents: delegating a review to a read-only specialist |
| 7 | Observability: the audit log as a table |

All model calls go through the team's **LiteLLM gateway**, the same one the rest of
the repo uses, so **no Anthropic API key is needed**.

---

## Quickstart

> **Python 3.10+ required.** Run these from the **repo root**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # includes claude-agent-sdk, pandas, ipykernel, notebook
cp .env.example .env                  # then fill in the LiteLLM values (see below)
```

Open the notebook either way:

- **VS Code:** open `agent-harness/claude_agent_harness.ipynb` and pick the `.venv` kernel.
- **Browser:** `jupyter notebook agent-harness/claude_agent_harness.ipynb`

Run the cells top to bottom. The **Check the gateway** cell in the setup section tells you
straight away if the URL, key or model name is wrong.

No Node.js needed: `claude-agent-sdk` bundles the Claude Code CLI it drives.

---

## Configuration (`.env`)

The notebook reads the repo-root `.env`, the same file the RAG exercises use:

| Variable | Used for |
|---|---|
| `LITELLM_PROXY_API_BASE` | Gateway URL, e.g. `https://your-litellm-proxy.example.com` |
| `LITELLM_PROXY_API_KEY` | Your gateway key (`sk-...`) |
| `LLM_MODEL` | Model the agent runs on, e.g. `litellm_proxy/anthropic/claude-sonnet-5`. Keep it a **Claude Sonnet** model. The notebook strips the `litellm_proxy/` prefix, and the same value still works for `src/rag.py`. |

To see which model names your gateway serves:

```bash
curl $LITELLM_PROXY_API_BASE/v1/models -H "Authorization: Bearer $LITELLM_PROXY_API_KEY"
```

After changing `.env`, **restart the kernel** so the new values are picked up.

---

## Try it yourself

- Add a new `@tool` (for example, a fake CRM lookup) and ask a question that needs it.
- Add a pattern to `BLOCKED` and try to get the agent around it.
- Remove `"Write"` from `allowed_tools` and watch how the agent copes when it can't save the report.
- Lower `max_turns` to 3 and watch the loop get cut short.
- Change `LLM_MODEL` to another gateway model, restart the kernel and compare.

**Docs:** [Agent SDK overview](https://docs.claude.com/en/docs/agent-sdk/overview) ·
[SDK repo](https://github.com/anthropics/claude-agent-sdk-python) ·
[LiteLLM + Claude Code](https://docs.litellm.ai/docs/tutorials/claude_responses_api)
