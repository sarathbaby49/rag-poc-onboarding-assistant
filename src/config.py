"""Central configuration for the Onboarding Assistant.

Everything tunable lives here so the session can point at one file and say
"this is the dial box." Change a value, re-run ingest, see the effect.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env so config picks up the model + proxy settings

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "sample_company"
CHROMA_DIR = BASE_DIR / ".chroma"          # local, on-disk vector store
COLLECTION_NAME = "onboarding"

# --- Chunking (Layer 2/3) ----------------------------------------------------
# Chunk size is the single biggest retrieval-quality lever. Too big = noisy
# context; too small = the answer is split across chunks and never retrieved.
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 120         # characters shared between neighbours (keeps context)

# --- Embeddings (Layer 3) ----------------------------------------------------
# Self-hosted, free, runs offline after first download. This is the "encoder"
# that turns text into vectors so we can compare meaning, not just keywords.
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for a demo
# CPU is plenty for this small model, and current CUDA builds of torch crash on
# older laptop GPUs (cudaErrorNoKernelImageForDevice) instead of falling back.
EMBED_DEVICE = "cpu"

# --- Retrieval (Layer 3) -----------------------------------------------------
TOP_K = 4                   # how many chunks to feed the model as context

# --- Generation via the LiteLLM gateway --------------------------------------
# Every model call goes through a LiteLLM proxy (an OpenAI-style gateway), so no
# direct provider API key is needed — just the proxy base URL + key in .env.
# Switch models by changing LLM_MODEL; nothing else in the code changes.
#   list models: curl $LITELLM_PROXY_API_BASE/v1/models -H "Authorization: Bearer $LITELLM_PROXY_API_KEY"
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL") or "litellm_proxy/openai/gpt-4.1-mini"
LITELLM_PROXY_API_BASE = os.getenv("LITELLM_PROXY_API_BASE", "")
LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY", "")
MAX_TOKENS = 1024

# Back-compat alias — older stubs referenced ANSWER_MODEL.
ANSWER_MODEL = LLM_MODEL

# --- Model tiers (Layer 5 — model selection & cost) ---------------------------
# Three tiers behind the same gateway. Routing exercises pick between them.
# Use the names your proxy exposes (see the curl command above).
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "litellm_proxy/anthropic/claude-haiku-4-5")
MID_MODEL = os.getenv("MID_MODEL", "litellm_proxy/anthropic/claude-sonnet-5-5")
STRONG_MODEL = os.getenv("STRONG_MODEL", "litellm_proxy/anthropic/claude-opus-5-5")

# --- LangSmith (Layer 4 — required for observability) -------------------------
# LangSmith traces every LLM call, tool invocation, and LangGraph step. This is
# NOT optional — it's how you debug agents and understand what happened during a
# run. Set these in your .env:
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=lsv2-...
#   LANGSMITH_PROJECT=onboarding-assistant
LANGSMITH_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "onboarding-assistant")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
