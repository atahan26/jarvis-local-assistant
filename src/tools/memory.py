"""Long-term memory that survives restarts.

Not in spec §8.3; added because a voice assistant that forgets you every time the
process exits is annoying in a way no latency number captures.

Design decisions worth stating:

* **Facts are stored, not transcripts.** Keeping whole conversations would blow
  the 16k context and bury the useful parts. The model writes short third-person
  statements instead ("prefers the volume around 30").
* **Facts go in the system prompt, not the message history.** History is trimmed
  as a conversation grows (`Conversation._trim`), so anything stored there would
  quietly fall out. The system prompt is rebuilt every request and always there.
* **Deduplication is on normalised text**, because a model asked to remember the
  same thing twice will phrase it slightly differently each time.
* **There is a cap.** Everything ever mentioned is not memory, it is a leak: the
  facts are prepended to every single request, so unbounded growth is unbounded
  cost on every turn.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .registry import Tool, ToolError, ToolResult

MAX_FACTS = 60
MAX_FACT_CHARS = 200


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


@dataclass
class Fact:
    text: str
    created_at: float = field(default_factory=time.time)
    source: str = "user"

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self.created_at))


class MemoryStore:
    """A small JSON-backed set of facts about the user."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._facts: list[Fact] = []
        self._lock = threading.Lock()
        self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt memory file must not stop the assistant from starting.
            return
        facts = []
        for item in raw.get("facts", []):
            if isinstance(item, dict) and item.get("text"):
                facts.append(
                    Fact(
                        text=str(item["text"]),
                        created_at=float(item.get("created_at", time.time())),
                        source=str(item.get("source", "user")),
                    )
                )
        with self._lock:
            self._facts = facts[-MAX_FACTS:]

    def save(self) -> None:
        with self._lock:
            payload = {"version": 1, "facts": [asdict(f) for f in self._facts]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic, so a crash cannot truncate the file

    # -- access ------------------------------------------------------------

    def all(self) -> list[Fact]:
        with self._lock:
            return list(self._facts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)

    def remember(self, text: str, source: str = "user") -> tuple[bool, str]:
        text = " ".join(str(text).split())
        if not text:
            raise ToolError("nothing to remember")
        if len(text) > MAX_FACT_CHARS:
            text = text[:MAX_FACT_CHARS].rstrip() + "..."

        needle = _normalise(text)
        with self._lock:
            for existing in self._facts:
                if _normalise(existing.text) == needle:
                    return False, existing.text
            self._facts.append(Fact(text=text, source=source))
            dropped = max(0, len(self._facts) - MAX_FACTS)
            if dropped:
                self._facts = self._facts[dropped:]
        self.save()
        return True, text

    def forget(self, query: str) -> list[str]:
        needle = _normalise(query)
        if not needle:
            raise ToolError("say what to forget")
        with self._lock:
            keep, removed = [], []
            for fact in self._facts:
                if needle in _normalise(fact.text):
                    removed.append(fact.text)
                else:
                    keep.append(fact)
            self._facts = keep
        if removed:
            self.save()
        return removed

    def forget_all(self) -> int:
        with self._lock:
            count = len(self._facts)
            self._facts = []
        self.save()
        return count

    def search(self, query: str) -> list[Fact]:
        needle = _normalise(query)
        if not needle:
            return self.all()
        words = [w for w in needle.split() if len(w) > 2]
        hits = []
        for fact in self.all():
            haystack = _normalise(fact.text)
            if needle in haystack or any(w in haystack for w in words):
                hits.append(fact)
        return hits

    # -- prompt ------------------------------------------------------------

    def as_prompt(self) -> str:
        """The block prepended to the system prompt each request."""
        facts = self.all()
        if not facts:
            return ""
        lines = "\n".join(f"- {f.text}" for f in facts)
        return (
            "\n\nWhat you remember about this user from previous conversations:\n"
            f"{lines}\n"
            "Use these naturally when relevant. Do not recite them back unprompted, "
            "and do not claim to remember anything that is not on this list."
        )


def make_memory_tool(store: MemoryStore):
    def memory(
        action: str, text: str | None = None, facts: list[str] | None = None
    ) -> ToolResult:
        if action == "remember":
            # One utterance often carries several facts ("my name is X and I
            # like Y"). Asked for a single string the model picked one and
            # dropped the other, then claimed to have stored both.
            items = [f for f in (facts or []) if str(f).strip()]
            if text and text.strip():
                items.append(text)
            if not items:
                raise ToolError("remember needs something to store")

            stored, duplicates = [], 0
            for item in items:
                added, kept = store.remember(str(item))
                if added:
                    stored.append(kept)
                else:
                    duplicates += 1

            if not stored:
                return ToolResult(True, "I already knew that.", {"duplicates": duplicates})
            return ToolResult(
                True,
                "Got it, I'll remember that." if len(stored) == 1
                else f"Got it, I'll remember those {len(stored)} things.",
                {"stored": stored, "duplicates": duplicates, "total": len(store)},
            )

        if action == "recall":
            hits = store.search(text or "")
            if not hits:
                return ToolResult(True, "I don't have anything about that.", {"facts": []})
            spoken = "; ".join(f.text for f in hits[:6])
            return ToolResult(True, f"Here's what I remember: {spoken}",
                              {"facts": [f.text for f in hits]})

        if action == "forget":
            if not text:
                raise ToolError("say what to forget")
            removed = store.forget(text)
            if not removed:
                return ToolResult(False, "I didn't have anything about that.")
            return ToolResult(True, f"Forgotten. I dropped {len(removed)} thing"
                                    + ("s." if len(removed) != 1 else "."),
                              {"removed": removed})

        if action == "forget_all":
            count = store.forget_all()
            return ToolResult(True, f"Cleared everything I remembered, {count} thing"
                                    + ("s." if count != 1 else "."))

        if action == "list":
            facts = store.all()
            if not facts:
                return ToolResult(True, "I don't remember anything yet.", {"facts": []})
            spoken = "; ".join(f.text for f in facts[:10])
            more = "" if len(facts) <= 10 else f", and {len(facts) - 10} more"
            return ToolResult(True, f"I remember: {spoken}{more}.",
                              {"facts": [f.text for f in facts]})

        raise ToolError(f"unknown action {action!r}")

    return memory


def build_tool(store: MemoryStore) -> Tool:
    return Tool(
        name="memory",
        description=(
            "Remember things about the user across conversations, or recall and "
            "forget them. Store a fact whenever the user says something durable "
            "about themselves, their name, their preferences or their setup, or "
            "asks you to remember it. Write each as a short third-person "
            "statement, for example 'is called Atahan' or 'prefers the volume "
            "around thirty'. If the user mentions several things at once, put "
            "EVERY one of them in the facts list; do not pick just one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "recall", "forget", "forget_all", "list"],
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For 'remember': one entry per distinct fact. Use this "
                        "whenever the user mentions more than one thing."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "A single fact to remember, or what to search for / forget."
                    ),
                },
            },
            "required": ["action"],
        },
        handler=make_memory_tool(store),
        # Wiping everything is destructive and irreversible.
        confirm_actions=("forget_all",),
    )
