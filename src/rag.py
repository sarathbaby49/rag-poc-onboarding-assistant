"""Layer 3 — RAG answer generation, the "G" in RAG  (YOUR EXERCISES: G1–G4).

    retrieve relevant chunks  ->  build a grounded prompt  ->  ask the model  ->  cite sources

This is the payoff of the whole repo: Ingestion + Embeddings + Retrieval + Memory
all exist to get the *right context* in front of the model here, so it can give a
grounded, cited answer — and honestly say "I don't know" when the context is thin.

  - G1 (core):    implement `answer`               -> straight retrieve → generate → cite
  - G2 (core):    implement `answer_or_abstain`     -> refuse to answer off-topic questions
  - G3 (stretch): implement `used_sources`          -> return only the sources actually cited
  - G4 (stretch): implement `answer_conversational` -> thread session memory for follow-ups
  - G5 (explore): tune SYSTEM_PROMPT below          -> the line that makes RAG *safe*

The scaffolding is done for you: SYSTEM_PROMPT (the assistant's "constitution") and
format_context (numbers the chunks so the model can cite them) — just like
_encoder()/_collection() were handed to you in the Retrieval exercises.

Self-contained: these G exercises DON'T depend on the earlier sessions being
finished. `semantic_search` (R1), `confident_hits` (R3) and `SessionMemory` (M1)
are provided ready-made by src/rag_helper.py (see the import below), so you can
build the whole generation layer on its own. (If you'd rather run on your OWN
R1/R3/M1, import them from src.retrieve / src.memory instead.)

Reference answers: solutions/rag.py.  Verify the gateway first: python -m checks.check_llm
Self-check your work:  python -m checks.check_rag
"""

from __future__ import annotations

import re
import sys

from src import config
from src.llm import complete
# semantic_search (R1) and confident_hits (R3) come from rag_helper, not
# src.retrieve, so these G exercises are self-contained — they run even if the
# retrieval exercises aren't done yet. See src/rag_helper.py.
from src.rag_helper import semantic_search, confident_hits

# The system prompt is the assistant's "constitution". Note the two rules that
# make RAG safe: answer only from context, and say so when the context is thin.
# (G5: this is the single highest-leverage line of the whole layer — experiment!)
SYSTEM_PROMPT = """You are the Engineering Onboarding Assistant for a software team.
You help new joinees get productive: setup, architecture, codebase and process questions.

Rules:
- Answer ONLY using the numbered context sources provided by the user.
- Cite the sources you used inline like [1], [2] after the relevant sentence.
- If the context does not contain the answer, say so plainly and suggest who to ask.
- Be concise and practical. Prefer steps and commands over prose."""

# What the assistant says when nothing relevant was retrieved (used by G2). It hands
# off to a real human channel from the corpus instead of guessing.
ABSTAIN_MESSAGE = (
    "I couldn't find this in the onboarding docs. "
    "Try asking in #eng-help on Slack, or check with your onboarding buddy."
)


def format_context(hits: list[dict]) -> str:
    """Turn retrieved chunks into a numbered block the model can cite by number.

    Done for you. Produces:
        [1] (source: data/sample_company/setup.md)
        <chunk text>

        [2] (source: ...)
        ...
    """
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(lines)


# --- G1: grounded generation -------------------------------------------------
def answer(question: str, k: int = config.TOP_K) -> dict:
    """EXERCISE G1 — the core RAG loop. Return {"answer", "sources"}.

    The whole idea of RAG: don't ask the model what it *knows*, hand it the
    retrieved context and make it answer ONLY from that, citing where each fact
    came from. That's what makes answers trustworthy and checkable.

    Steps:
      1. hits = semantic_search(question, k)          # provided by rag_helper
      2. context = format_context(hits)
      3. Ask the model through the gateway, giving it the SYSTEM_PROMPT plus the
         numbered context and the question:
             text = complete(
                 [
                     {"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",
                      "content": f"Context sources:\\n\\n{context}\\n\\nQuestion: {question}"},
                 ],
                 max_tokens=config.MAX_TOKENS,
             )
      4. return {"answer": text, "sources": hits}

    Self-check:  python -m checks.check_rag   (live check needs the gateway)
    """
    # TODO(G1): implement the four steps above. Reference: solutions/rag.py.
    hits = semantic_search(question, k)
    text = _generate(question, hits)
    return {"answer": text, "sources": hits}


# --- G2: honest "I don't know" ----------------------------------------------
def answer_or_abstain(question: str, k: int = config.TOP_K, min_score: float = 0.25) -> dict:
    """EXERCISE G2 — the shippable version: refuse to answer off-topic questions.

    Why: `semantic_search` ALWAYS returns k chunks, even for a question the corpus
    can't answer — so a naive assistant will happily answer from junk. Gate on
    confidence: if nothing clears the bar, hand off to a human WITHOUT calling the
    model (no tokens spent, zero chance of a hallucinated answer).

    Steps:
      1. hits = confident_hits(question, k, min_score)      # provided by rag_helper
      2. if not hits:  return {"answer": ABSTAIN_MESSAGE, "sources": []}   # short-circuit!
      3. otherwise, generate exactly like G1 over `hits` and return {"answer", "sources"}.

    Self-check:  python -m checks.check_rag   (the abstain path is checkable with NO key)
    """
    # TODO(G2): confidence-gate, then either abstain or generate.
    hits = confident_hits(question, k, min_score)  # R3: drops weak matches
    if not hits:
        # Nothing cleared the bar — hand off without spending a model call.
        return {"answer": ABSTAIN_MESSAGE, "sources": []}
    text = _generate(question, hits)
    return {"answer": text, "sources": used_sources(text, hits) or hits}


# --- G3: trustworthy citations ----------------------------------------------
def used_sources(answer_text: str, hits: list[dict]) -> list[dict]:
    """EXERCISE G3 — return only the sources the model actually cited.

    Grounding you don't verify is just a promise. You hand the model k sources but
    it may only cite [2] — so the "Sources" panel should show [2], not all k. This
    is a pure function (no model call), so it self-checks with NO key.

    Steps:
      1. Find every [n] marker in `answer_text`  (regex: r"\\[(\\d+)\\]").
      2. Return the hits whose 1-based position is in that set (keep original order).

    Examples:
      used_sources("See [1] and [3].", hits3)  -> [hits3[0], hits3[2]]
      used_sources("No citations here.", hits3) -> []

    Wire it into answer()/answer_or_abstain() so sources = used_sources(text, hits).

    Self-check:  python -m checks.check_rag
    """
    # TODO(G3): parse the [n] markers and filter hits. One regex + one comprehension.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    return [h for i, h in enumerate(hits, start=1) if i in cited]


# --- G4: conversational RAG (memory) ----------------------------------------
def answer_conversational(question: str, memory, k: int = config.TOP_K) -> dict:
    """EXERCISE G4 — thread session memory so follow-ups work.

    "how do I run it?"  ->  "what about the tests?" only makes sense if the model
    sees the last few turns. `memory` is a SessionMemory (provided ready-made by
    rag_helper); its `as_messages()` returns the recent window, so the token
    budget stays bounded.

    Steps:
      1. hits = semantic_search(question, k);  context = format_context(hits)
      2. messages = [
             {"role": "system", "content": SYSTEM_PROMPT},
             *memory.as_messages(),                        # the recent window (M1)
             {"role": "user",
              "content": f"Context sources:\\n\\n{context}\\n\\nQuestion: {question}"},
         ]
      3. text = complete(messages, max_tokens=config.MAX_TOKENS)
      4. record both turns:  memory.add("user", question); memory.add("assistant", text)
      5. return {"answer": text, "sources": hits}

    Self-check:  python -m checks.check_rag   (live check needs the gateway)
    """
    # TODO(G4): splice memory.as_messages() into the call, then record the turn.
    raise NotImplementedError("Exercise G4: implement answer_conversational — see EXERCISES.md (Part E)")


def _cli() -> None:
    question = " ".join(sys.argv[1:]) or "How do I set up my local environment?"
    result = answer(question)
    print(f"\nQ: {question}\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, hit in enumerate(result["sources"], start=1):
        print(f"[{i}] {hit['source']}  (score {hit['score']:.2f})")


def _generate(question: str, hits: list[dict]) -> str:
    """Shared generation step: hand the model the context + question, get text back."""
    context = format_context(hits)
    return complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context sources:\n\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=config.MAX_TOKENS,
    )

if __name__ == "__main__":
    _cli()
