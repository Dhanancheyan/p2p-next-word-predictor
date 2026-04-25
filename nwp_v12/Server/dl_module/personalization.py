"""
personalization.py
Lightweight on-device personalisation layer for the keyboard LM.

Components
----------
RecentWordCache    : Sliding window of recently used words for score boosting.
UserDictionary     : User-defined words, names, slang, and custom tokens with
                     per-entry importance weights.
PersonalizationLayer : Combines both components into a candidate re-ranker.

Usage
-----
    pers = PersonalizationLayer()
    pers.observe("hello world")
    ranked = pers.rerank(
        candidates=["hello", "help", "her"],
        base_scores=[0.9, 0.8, 0.7],
    )

Persistence
-----------
    pers.save("path/to/pers_<model_id>.json")
    pers.load("path/to/pers_<model_id>.json")
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Recent Word Cache
# ---------------------------------------------------------------------------

class RecentWordCache:
    """Sliding window of the N most recently used words."""

    def __init__(self, capacity: int = 128) -> None:
        self._window: deque[str] = deque(maxlen=capacity)
        self._freq: Counter[str] = Counter()

    def observe(self, word: str) -> None:
        """Record a single word into the recency window."""
        word = word.strip().lower()
        if not word:
            return
        if len(self._window) == self._window.maxlen:
            oldest = self._window[0]
            self._freq[oldest] = max(0, self._freq[oldest] - 1)
        self._window.append(word)
        self._freq[word] += 1

    def score(self, word: str) -> float:
        """Return a normalised recency score in [0, 1]."""
        c = self._freq.get(word.strip().lower(), 0)
        if not c:
            return 0.0
        max_c = max(self._freq.values()) if self._freq else 1
        return float(c) / max(max_c, 1)

    def top_words(self, k: int = 10) -> list[str]:
        """Return the k most frequently used words in the window."""
        return [w for w, _ in self._freq.most_common(k)]


# ---------------------------------------------------------------------------
# User Dictionary
# ---------------------------------------------------------------------------

@dataclass
class UserDictEntry:
    word: str
    weight: float = 1.0        # user-defined importance, clamped to [0.5, 2.0]
    category: str = "custom"   # "name", "slang", "custom", or "technical"
    created_at: float = field(default_factory=time.time)
    use_count: int = 0


class UserDictionary:
    """Persistent user-defined vocabulary with per-entry weights."""

    def __init__(self) -> None:
        self._entries: dict[str, UserDictEntry] = {}  # lowercase key -> entry

    def add(
        self,
        word: str,
        *,
        weight: float = 1.0,
        category: str = "custom",
    ) -> None:
        """Add or replace a word in the dictionary."""
        key = word.strip().lower()
        if key:
            self._entries[key] = UserDictEntry(
                word=word.strip(), weight=weight, category=category
            )

    def remove(self, word: str) -> None:
        """Remove a word from the dictionary (no-op if absent)."""
        self._entries.pop(word.strip().lower(), None)

    def contains(self, word: str) -> bool:
        return word.strip().lower() in self._entries

    def score(self, word: str) -> float:
        """Return the entry weight and increment use_count, or 0.0 if absent."""
        entry = self._entries.get(word.strip().lower())
        if entry is None:
            return 0.0
        entry.use_count += 1
        return float(entry.weight)

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise all entries for the /local/personalization/words endpoint."""
        return [asdict(e) for e in self._entries.values()]

    def from_list(self, data: list[dict[str, Any]]) -> None:
        """Restore entries from a serialised list."""
        for item in data:
            key = item.get("word", "").strip().lower()
            if key:
                self._entries[key] = UserDictEntry(**item)


# ---------------------------------------------------------------------------
# Personalisation Layer
# ---------------------------------------------------------------------------

class PersonalizationLayer:
    """
    Re-ranks model suggestions using recency and user dictionary signals.

    Scoring formula:
        final_score = base_score + ALPHA * recency_score + BETA * user_dict_score

    ALPHA and BETA are deliberately small so personalisation nudges rather than
    overrides the base model probabilities.
    """

    ALPHA: float = 0.15   # recency boost weight
    BETA: float = 0.20    # user dictionary boost weight

    def __init__(self, cache_capacity: int = 256) -> None:
        self.cache = RecentWordCache(capacity=cache_capacity)
        self.user_dict = UserDictionary()

    def observe(self, text: str) -> None:
        """Record words from accepted or typed text into the recency cache."""
        for word in text.split():
            self.cache.observe(word)

    def rerank(
        self,
        candidates: list[str],
        base_scores: list[float],
    ) -> list[tuple[str, float]]:
        """
        Return candidates sorted by combined score (descending).

        Parameters
        ----------
        candidates   : Candidate words from the base model.
        base_scores  : Corresponding base model probabilities.

        Returns a list of (word, combined_score) tuples.
        """
        results: list[tuple[str, float]] = []
        for word, base in zip(candidates, base_scores):
            rec = self.cache.score(word)
            ud = self.user_dict.score(word)
            combined = base + self.ALPHA * rec + self.BETA * ud
            results.append((word, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def top_personal_suggestions(self, k: int = 3) -> list[str]:
        """Return the k most frequently used words, independent of the base LM."""
        return self.cache.top_words(k=k)

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist recency window and user dictionary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "cache_window": list(self.cache._window),
            "user_dict": self.user_dict.to_list(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load state from a JSON file produced by save(). Silent on missing file."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for w in data.get("cache_window", []):
                self.cache.observe(w)
            self.user_dict.from_list(data.get("user_dict", []))
        except Exception:
            pass
