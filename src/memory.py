"""Layer 3 — Memory  (YOUR EXERCISES: M1 + M3 + M4).

Two kinds of memory make an onboarding assistant feel personal:

1. Session memory  — remembers the last few turns so follow-ups work
   ("how do I run it?"  ->  "what about the tests?" without repeating context).
2. Long-term memory — remembers WHO the joinee is (role, team, progress) across
   days, so it can say "you finished setup; next is your first PR".

Hands-on exercises:

  - M1: implement `SessionMemory.as_messages`     (windowed short-term memory)
  - M3: implement `SummaryBufferMemory.add`       (compress overflow into a
        running summary so old context is kept as a gist, not dropped — the
        answer to "a plain window forgets everything that scrolls off").
  - M4: implement `MemoryStore.recall`            (retrieve the most RELEVANT
        past memory by meaning, not just the most recent — memory as retrieval;
        the answer to "when context length alone isn't enough").

`JoineeProfile` (persistent long-term memory of role + onboarding progress) is
provided, already working — read it as the reference example, no code needed.

Self-check your work:  python -m checks.check_memory
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src import config


@dataclass
class SessionMemory:
    """Short-term: the current conversation."""

    window: int = 6                                   # how many recent turns to keep
    turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def as_messages(self) -> list[dict]:
        """EXERCISE M1 — return only the last `self.window` turns, oldest first.

        Why: prepending the whole history to every call blows the token budget.
        A sliding window keeps follow-ups working cheaply.

        Hint: a list slice does it in one line ( turns[-window:] ).

        Self-check:  python -m checks.check_memory
        """
        return self.turns[-self.window:]


# ---------------------------------------------------------------------------
# PROVIDED (not an exercise) — long-term memory of a joinee's role + progress.
# Read it as the worked example of persistent long-term memory; M4 below is the
# exercise that goes further (recall by meaning).
# ---------------------------------------------------------------------------
PROFILE_DIR = config.BASE_DIR / ".profiles"


@dataclass
class JoineeProfile:
    """Long-term: who this person is and how far they've got (persisted to disk)."""

    name: str
    role: str = "engineer"
    completed_steps: list[str] = field(default_factory=list)

    def record_step(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.save()  # persist immediately so progress survives a restart

    def _path(self) -> Path:
        return PROFILE_DIR / f"{self.name}.json"

    def save(self) -> None:
        """Persist name / role / completed_steps as JSON so it survives a restart."""
        PROFILE_DIR.mkdir(exist_ok=True)
        self._path().write_text(
            json.dumps(
                {"name": self.name, "role": self.role, "completed_steps": self.completed_steps},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, name: str) -> "JoineeProfile":
        """Load a previously saved profile back from disk."""
        data = json.loads((PROFILE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            role=data.get("role", "engineer"),
            completed_steps=data.get("completed_steps", []),
        )


# ---------------------------------------------------------------------------
# M3 — SummaryBufferMemory: a window that COMPRESSES overflow instead of
# dropping it. Still short-term (lives in memory, scoped to the conversation),
# but old turns that scroll off the window are folded into a running summary so
# their gist survives. This is the "compress" fix for when context length alone
# isn't enough (the "retrieve" fix is M4 below).
# ---------------------------------------------------------------------------
def _extractive_summary(previous: str, role: str, content: str) -> str:
    """Provided — the default summariser: append the turn as a bullet line.

    "Extractive" because it just extracts/concatenates the text rather than
    paraphrasing it. In a real app you'd swap this for an LLM call, but a plain
    function keeps the exercise API-key-free and deterministic to self-check.
    """
    line = f"- {role}: {content}".strip()
    return f"{previous}\n{line}".strip() if previous else line


@dataclass
class SummaryBufferMemory:
    """Short-term memory: a sliding window that compresses overflow into a summary.

    Like `SessionMemory`, but instead of hard-dropping turns older than the
    window, it folds them into `self.summary` via the pluggable `summarize`
    callable. The recent turns stay verbatim; everything older survives as a gist.
    """

    window: int = 4
    turns: list[dict] = field(default_factory=list)
    summary: str = ""
    summarize: Callable[[str, str, str], str] = _extractive_summary

    def add(self, role: str, content: str) -> None:
        """EXERCISE M3 — append a turn, then compress any overflow into the summary.

        Steps:
          1. append `{"role": role, "content": content}` to `self.turns`.
          2. while there are more than `self.window` turns, pop the OLDEST one
             (`self.turns.pop(0)`) and fold it into `self.summary`:
                 self.summary = self.summarize(self.summary, old["role"], old["content"])

        Why: a plain window forgets everything that scrolls off. Compressing the
        overflow keeps the gist of the whole conversation for a fixed token cost.

        Self-check:  python -m checks.check_memory
        """
        self.turns.append({"role": role, "content": content})
        while len(self.turns) > self.window:
            old = self.turns.pop(0)
            self.summary = self.summarize(self.summary, old["role"], old["content"])

    def as_context(self) -> list[dict]:
        """Provided — assemble what you'd feed the model: summary + live turns.

        The running summary goes first as a `system` message (only if non-empty),
        followed by the verbatim recent turns.
        """
        msgs: list[dict] = []
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation so far:\n{self.summary}"})
        return msgs + self.turns


# ---------------------------------------------------------------------------
# M4 — MemoryStore: semantic recall (short-term recency vs long-term relevance)
# ---------------------------------------------------------------------------
@dataclass
class MemoryStore:
    """Long-term memory you search by MEANING, not by recency.

    A window returns the *latest* turns. But the fact that answers the current
    question may have been mentioned 50 turns ago ("I'm on the payments team").
    So we embed every memory once and, at recall time, return the few that are
    closest in meaning to the query — exactly the retrieval you built in R1,
    pointed at the conversation instead of the docs.
    """

    # each item: {"text": str, "embedding": list[float]}
    items: list[dict] = field(default_factory=list)

    def remember(self, text: str) -> None:
        """Provided — embed a memory once and keep it for later recall."""
        from src.embeddings import embed

        self.items.append({"text": text, "embedding": embed(text)})

    def recall(self, query: str, k: int = 3) -> list[str]:
        """EXERCISE M4 — return the `k` most RELEVANT remembered texts.

        Steps:
          1. embed the query:  `from src.embeddings import embed; q = embed(query)`
          2. score every item by similarity to `q`. Embeddings from `embed` are
             normalised, so cosine similarity is just the dot product:
                 sum(a * b for a, b in zip(q, item["embedding"]))
          3. return the `text` of the top `k`, most similar first.

        Note this is recall by *relevance*, not recency — the whole point. The
        newest memory is not necessarily the one that comes back.

        Self-check:  python -m checks.check_memory
        """
        from src.embeddings import embed

        if not self.items:
            return []
        q = embed(query)
        scored = [
            (sum(a * b for a, b in zip(q, item["embedding"])), item["text"])
            for item in self.items
        ]
        scored.sort(key=lambda pair: -pair[0])
        return [text for _, text in scored[:k]]

    # --- persistence (provided) — makes long-term memory durable, like JoineeProfile.
    #     We store the embeddings too, so we never have to re-embed on load.
    def save(self, path: Path) -> None:
        """Write all remembered items (text + embedding) to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.items), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MemoryStore":
        """Load a saved store back from disk; a fresh empty store if none exists."""
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            items = []
        return cls(items=items if isinstance(items, list) else [])
