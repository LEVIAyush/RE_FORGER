"""
Lightweight persistent memory for the research agent.

The previous implementation (`self.memory = {}` inside ResearchAgent) was
reset at the start of every single `run()` call and only ever populated by
hardcoded regex for three countries' population/area — so it was neither
persistent nor general-purpose, despite being described as the project's
"memory system".

This version stores every (topic, step, result) finding to a JSON file on
disk, so it survives across questions AND across app restarts, and works
for any topic rather than a hardcoded list. Recall uses simple keyword
overlap scoring — no extra ML dependency required, which keeps the project
on free, lightweight tools as intended.
"""
import json
import time
from pathlib import Path

DEFAULT_MEMORY_FILE = Path(__file__).resolve().parent.parent / "data" / "memory_store.json"
MAX_ENTRIES = 500  # bound unbounded growth


class MemoryStore:
    def __init__(self, path: Path = DEFAULT_MEMORY_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> list:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"  -> Warning: could not persist memory ({e})")

    # -- public API --------------------------------------------------------
    def add(self, topic: str, step: str, result: str):
        """Store one finding. Called after every executed research step."""
        self._entries.append(
            {"topic": topic, "step": step, "result": result, "timestamp": time.time()}
        )
        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[-MAX_ENTRIES:]
        self._save()

    def recall(self, query: str, k: int = 4) -> list:
        """
        Return the k findings most relevant to `query`, by simple keyword
        overlap. Falls back to the k most recent entries if nothing matches,
        so a follow-up question still gets *some* context.
        """
        if not self._entries:
            return []

        query_words = {w for w in query.lower().split() if len(w) > 2}

        def score(entry):
            text = f"{entry['topic']} {entry['step']} {entry['result']}".lower()
            return sum(1 for w in query_words if w in text)

        scored = [(score(e), e) for e in self._entries]
        matched = sorted((e for s, e in scored if s > 0), key=lambda e: score(e), reverse=True)

        if matched:
            return matched[:k]
        return self._entries[-k:]  # fallback: most recent, so follow-ups still have context

    def all_entries(self) -> list:
        return list(self._entries)

    def clear(self):
        """Wipe all persisted memory (used only if the user explicitly asks)."""
        self._entries = []
        self._save()
