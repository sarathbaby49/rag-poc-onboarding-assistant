"""Layer 3 — Memory  (YOUR EXERCISES: M1 + M2).

Two kinds of memory make an onboarding assistant feel personal:

1. Session memory  — remembers the last few turns so follow-ups work
   ("how do I run it?"  ->  "what about the tests?" without repeating context).
2. Long-term memory — remembers WHO the joinee is (role, team, progress) across
   days, so it can say "you finished setup; next is your first PR".

  - M1: implement `SessionMemory.as_messages`     (windowed short-term memory)
  - M2: implement `JoineeProfile.save` / `.load`  (persistent long-term memory)

Self-check your work:  python -m checks.check_memory
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        # TODO(M1): return the last `self.window` turns instead of everything.
        raise NotImplementedError("Exercise M1: implement as_messages — see EXERCISES.md")


@dataclass
class JoineeProfile:
    """Long-term: who this person is and how far they've got."""

    name: str
    role: str = "engineer"
    completed_steps: list[str] = field(default_factory=list)

    def record_step(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        # TODO(M2): once save() works, call self.save() here so progress persists.

    def save(self) -> None:
        """EXERCISE M2 — persist this profile so it survives a restart.

        Write name / role / completed_steps as JSON to a file named after the
        joinee (e.g. BASE_DIR/.profiles/<name>.json). Create the folder if needed.

        Hint: `from src import config` gives you config.BASE_DIR;
              json.dumps(...) + Path.write_text(...) is all you need.

        Self-check:  python -m checks.check_memory
        """
        # TODO(M2): write this profile to a JSON file on disk.
        raise NotImplementedError("Exercise M2: implement save — see EXERCISES.md")

    @classmethod
    def load(cls, name: str) -> "JoineeProfile":
        """EXERCISE M2 — load a previously saved profile back from disk.

        Read the JSON file save() wrote for `name` and return a JoineeProfile
        with the same name, role and completed_steps.
        """
        # TODO(M2): read the JSON file and return a JoineeProfile.
        raise NotImplementedError("Exercise M2: implement load — see EXERCISES.md")
