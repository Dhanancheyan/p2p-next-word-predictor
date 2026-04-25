"""
cache_agent.py
Session-level phrase cache for sub-millisecond next-word prediction.

The cache is purely in-memory and resets each server run.  It provides two
complementary layers:

1. Phrase cache  : Maps recent (context -> next_word) pairs learned from text
                   the user has typed this session.  Lookups try trigram context
                   first, then bigram.

2. Word frequency: Sliding window of recently accepted words used for recency
                   scoring in the personalisation layer.

Cache predictions are intended to be checked before model inference and can
short-circuit the LSTM/n-gram pipeline for recently repeated phrases.
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict, deque
from typing import Any

_WORD_RE = re.compile(r"[a-zA-Z']+|[0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t]


class CacheAgent:
    """
    Fast session-level cache for next-word prediction.

    Parameters
    ----------
    phrase_capacity : Maximum number of distinct context keys to store.
    word_capacity   : Sliding window size for the word frequency tracker.
    """

    PHRASE_CAPACITY = 512
    WORD_CAPACITY = 256

    def __init__(
        self,
        phrase_capacity: int = PHRASE_CAPACITY,
        word_capacity: int = WORD_CAPACITY,
    ) -> None:
        # LRU phrase cache: context_key -> Counter of next_words.
        self._phrase: OrderedDict[str, Counter] = OrderedDict()
        self._phrase_capacity = phrase_capacity
        # Sliding word window for recency scoring.
        self._word_window: deque[str] = deque(maxlen=word_capacity)
        self._word_freq: Counter[str] = Counter()

    # -- Observation ----------------------------------------------------------

    def observe(self, text: str) -> None:
        """Learn bigram and trigram transitions from a block of typed text."""
        words = _tokenize(text)
        for word in words:
            self._observe_word(word)
        for i in range(1, len(words)):
            self._add_phrase([words[i - 1]], words[i])
            if i >= 2:
                self._add_phrase([words[i - 2], words[i - 1]], words[i])

    def _observe_word(self, word: str) -> None:
        if len(self._word_window) == self._word_window.maxlen:
            oldest = self._word_window[0]
            self._word_freq[oldest] = max(0, self._word_freq[oldest] - 1)
        self._word_window.append(word)
        self._word_freq[word] += 1

    def _add_phrase(self, context: list[str], next_word: str) -> None:
        key = " ".join(context)
        if key in self._phrase:
            self._phrase.move_to_end(key)
        else:
            if len(self._phrase) >= self._phrase_capacity:
                self._phrase.popitem(last=False)
            self._phrase[key] = Counter()
        self._phrase[key][next_word] += 1

    # -- Lookup ---------------------------------------------------------------

    def lookup(self, words: list[str], k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k (word, score) pairs from the phrase cache.

        Tries trigram context first then falls back to bigram.
        Returns an empty list on a cache miss.
        """
        if not words:
            return []

        # Trigram context.
        if len(words) >= 2:
            result = self._top_from_counter(self._phrase.get(" ".join(words[-2:])), k)
            if result:
                return result

        # Bigram (single-word) context.
        return self._top_from_counter(self._phrase.get(words[-1]), k)

    def _top_from_counter(
        self, counter: Counter | None, k: int
    ) -> list[tuple[str, float]]:
        if not counter:
            return []
        total = sum(counter.values())
        if total == 0:
            return []
        return [(w, c / total) for w, c in counter.most_common(k)]

    # -- Recency score --------------------------------------------------------

    def recency_score(self, word: str) -> float:
        """Return a normalised recency score in [0, 1] for a single word."""
        c = self._word_freq.get(word.lower(), 0)
        if not c:
            return 0.0
        max_c = max(self._word_freq.values()) if self._word_freq else 1
        return float(c) / max(max_c, 1)

    def top_words(self, k: int = 5) -> list[str]:
        """Return the k most frequently used words this session."""
        return [w for w, _ in self._word_freq.most_common(k)]

    # -- Diagnostics ----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a summary dict for the /local/metrics endpoint."""
        return {
            "phrase_keys": len(self._phrase),
            "word_window_size": len(self._word_window),
            "unique_words": len(self._word_freq),
        }
