"""REFERENCE SOLUTION for src/memory.py (Exercises M1 + M2).

Try the exercise first! If you're stuck or out of time, copy the relevant
body into src/memory.py. Don't peek before you've had a go.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src import config

PROFILE_DIR = config.BASE_DIR / ".profiles"


@dataclass
class SessionMemory:
    """Short-term: the current conversation, capped to the last `window` turns."""

    window: int = 6
    turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    # --- M1 ---
    def as_messages(self) -> list[dict]:
        # Keep only the most recent `window` turns (watch the token budget).
        return self.turns[-self.window:]


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

    # --- M2 ---
    def _path(self) -> Path:
        return PROFILE_DIR / f"{self.name}.json"

    def save(self) -> None:
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
        data = json.loads((PROFILE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            role=data.get("role", "engineer"),
            completed_steps=data.get("completed_steps", []),
        )
