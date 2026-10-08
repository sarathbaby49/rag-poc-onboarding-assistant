"""Layer 6 — the system under test (done for you, don't edit).

One function, `ask_assistant`, that the eval suite and the Production Lab both
call. It's the G2 pipeline from src/rag.py (confidence gate → grounded
generation), with two production additions:

  - a PROMPT VERSION, so every answer and trace says which prompt produced it
  - every model call goes through observability.call_llm, so it gets traced

Two prompt versions ship, to make the regression demo real:

  v1  the current production prompt: answer only from context, abstain when the
      retriever finds nothing relevant (min_score 0.25).
  v2  a "friendlier" prompt someone wants to ship because users complained about
      too many "I don't know"s: it lowers the gate to 0 and lets the model fall
      back on general knowledge. Run the eval suite to see what that costs.
"""

from __future__ import annotations

from src import config
from src.observability import call_llm
from src.rag import ABSTAIN_MESSAGE, SYSTEM_PROMPT, format_context
from src.rag_helper import confident_hits

PROMPTS = {
    "v1": {"system": SYSTEM_PROMPT, "min_score": 0.25},
    "v2": {
        "system": """You are a friendly Engineering Onboarding Assistant.
Use the numbered context sources when they help, citing them like [1].
If the sources don't cover the question, answer from your general knowledge so the
user is never left without an answer. Be confident and helpful.""",
        "min_score": 0.0,
    },
}


def ask_assistant(question: str, prompt_version: str = "v1", user_id: str = "demo-user",
                  k: int = config.TOP_K) -> dict:
    """Answer one question. Returns {answer, sources, abstained, usage}."""
    settings = PROMPTS[prompt_version]
    hits = confident_hits(question, k, settings["min_score"])
    if not hits:
        return {"answer": ABSTAIN_MESSAGE, "sources": [], "abstained": True, "usage": None}

    messages = [
        {"role": "system", "content": settings["system"]},
        {"role": "user", "content": f"Context sources:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    result = call_llm(messages, name="rag_answer", prompt_version=prompt_version,
                      user_id=user_id, max_tokens=config.MAX_TOKENS)
    return {"answer": result["text"], "sources": hits, "abstained": False, "usage": result}
