"""REFERENCE SOLUTION — Layer 5 — Prompt caching: static first, dynamic last.  (EXERCISE C4)

The provider can cache the processed START of a prompt. If the next request
begins with exactly the same text, those tokens are billed at ~10% of the input
price (Anthropic: write 1.25x once, then read 0.1x — see src/cost.py).

The cache matches from the very first token, so ONE changed character early in
the prompt (a timestamp, the user's name, the question) breaks it for everything
after. Rule: put what never changes first, what changes every call last.

Here the static part is the "team handbook": every doc and code file in
data/sample_company (a few thousand tokens). Small corpora like ours can simply
ship the whole handbook in a cached prefix.
"""

from __future__ import annotations

from functools import lru_cache

from src import config

SYSTEM_PROMPT = (
    "You are the Acme Shop onboarding assistant. Answer new engineers' questions "
    "using ONLY the team handbook and any extra context below. Cite the file names "
    "you used. If the answer isn't there, say you don't know."
)


@lru_cache(maxsize=1)
def handbook() -> str:
    """Every doc + code file, in a fixed order (done for you).

    Sorted so the text is byte-identical on every call — a requirement for caching.
    """
    root = config.DATA_DIR
    files = sorted(root.glob("*.md")) + sorted((root / "code").glob("*.py"))
    parts = [f"=== {p.relative_to(root)} ===\n{p.read_text()}" for p in files]
    return "\n\n".join(parts)


def format_hits(hits: list[dict] | None) -> str:
    """Retrieved chunks as a numbered block (done for you)."""
    if not hits:
        return "(none)"
    return "\n\n".join(f"[{i}] ({h['source']}) {h['text']}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[dict] | None = None) -> list[dict]:
    """Static, cached prefix first; dynamic context and question last."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT}\n\n{handbook()}",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": f"Extra context:\n{format_hits(hits)}\n\nQuestion: {question}",
        },
    ]
