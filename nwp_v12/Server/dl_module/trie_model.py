"""
trie_model.py
Trie-based n-gram language model (trigram / bigram / unigram with backoff).

Classes
-------
TrieNode   : Compact prefix-tree node.
NgramModel : Full trigram model with Laplace smoothing, autocomplete, and
             federated merge.  Vocabulary is bounded to max_vocab words;
             least-frequent entries are evicted when the cap is reached.

Functions
---------
tokenize(text)     : Lowercase alphanumeric + apostrophe tokeniser.
build_seed_model() : Return an NgramModel pre-seeded from Data/seed_corpus.txt.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z']+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser: alphanumeric tokens and apostrophes only."""
    return [t.lower() for t in _WORD_RE.findall(text) if t]


# ---------------------------------------------------------------------------
# Seed corpus loader
# ---------------------------------------------------------------------------

_SEED_CORPUS_PATH = os.path.join(
    os.path.dirname(  # dl_module/
        os.path.dirname(  # Server/
            os.path.dirname(os.path.abspath(__file__))  # nwp_v12/
        )
    ),
    "Data", "seed_corpus.txt",
)


def _load_seed_texts() -> list[str]:
    """
    Load seed sentences from Data/seed_corpus.txt.

    Lines starting with '#' and blank lines are skipped.
    Returns an empty list gracefully when the file is missing so the model
    still starts -- it will just have no prior knowledge until the user types.
    """
    try:
        with open(_SEED_CORPUS_PATH, encoding="utf-8") as f:
            return [
                ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Trie Node
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "count", "total")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.count: int = 0   # occurrences of this exact n-gram ending here
        self.total: int = 0   # sum of child counts (for probability normalisation)


# ---------------------------------------------------------------------------
# N-gram Model
# ---------------------------------------------------------------------------

class NgramModel:
    """
    Trigram language model stored as a nested trie.

    Supports:
    - observe(words)             : Record a token sequence.
    - predict(context_words, k)  : Top-k next-word predictions with trigram ->
                                   bigram -> unigram backoff.
    - autocomplete(prefix, k)    : Top-k completions for a partial word.
    - merge(other)               : Aggregate counts from a peer model (federated sum).
    - to_dict / from_dict        : JSON-serialisable persistence.

    Vocabulary is capped at max_vocab unique words (default 1 000).  When the
    cap is reached the least-frequent unigram is evicted before inserting a new
    word, keeping memory bounded on-device.
    """

    SMOOTHING = 0.5             # Laplace smoothing constant
    DEFAULT_MAX_VOCAB = 1000

    def __init__(self, max_ngram: int = 3, max_vocab: int = DEFAULT_MAX_VOCAB) -> None:
        self.max_ngram = max_ngram
        self.max_vocab = max_vocab
        self._root: TrieNode = TrieNode()
        self._unigram: dict[str, int] = defaultdict(int)
        self._total_unigrams: int = 0

    # -- Insertion ------------------------------------------------------------

    def observe(self, words: list[str]) -> None:
        """Record all n-grams from a token sequence, evicting rare words as needed."""
        for i, w in enumerate(words):
            self._maybe_evict(w)
            self._unigram[w] += 1
            self._total_unigrams += 1
            if i >= 1:
                self._insert([words[i - 1]], w, 1)
            if i >= 2:
                self._insert([words[i - 2], words[i - 1]], w, 1)

    def _maybe_evict(self, incoming: str) -> None:
        """Evict the least-frequent word if the vocabulary cap is reached."""
        if incoming in self._unigram:
            return
        if len(self._unigram) < self.max_vocab:
            return
        # Ties broken alphabetically for stability across runs.
        evict = min(self._unigram, key=lambda w: (self._unigram[w], w))
        evicted_count = self._unigram.pop(evict)
        self._total_unigrams -= evicted_count
        self._remove_word_from_trie(self._root, evict)

    def _remove_word_from_trie(self, node: TrieNode, word: str) -> None:
        """Remove word as a leaf child from every node in the trie (best-effort)."""
        if word in node.children:
            removed_count = node.children[word].count
            del node.children[word]
            node.total = max(0, node.total - removed_count)
        for child in node.children.values():
            self._remove_word_from_trie(child, word)

    def _insert(self, context: list[str], next_word: str, count: int) -> None:
        node = self._root
        for w in context:
            if w not in node.children:
                node.children[w] = TrieNode()
            node = node.children[w]
        if next_word not in node.children:
            node.children[next_word] = TrieNode()
        node.children[next_word].count += count
        node.total += count

    # -- Prediction -----------------------------------------------------------

    def predict(self, context_words: list[str], k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k (word, probability) pairs.

        Backoff order: trigram -> bigram -> unigram.
        Returns an empty list only when the model has no observations at all.
        """
        context = [w.lower() for w in context_words]

        if len(context) >= 2:
            result = self._lookup(context[-2:], k)
            if result:
                return result

        if len(context) >= 1:
            result = self._lookup(context[-1:], k)
            if result:
                return result

        return self._unigram_top(k)

    def _lookup(self, context: list[str], k: int) -> list[tuple[str, float]]:
        node = self._root
        for w in context:
            if w not in node.children:
                return []
            node = node.children[w]
        if not node.children:
            return []
        total = node.total + self.SMOOTHING * len(node.children)
        items = sorted(node.children.items(), key=lambda x: x[1].count, reverse=True)[:k]
        return [(word, (child.count + self.SMOOTHING) / total) for word, child in items]

    def _unigram_top(self, k: int) -> list[tuple[str, float]]:
        if not self._unigram:
            return []
        total = self._total_unigrams + self.SMOOTHING * len(self._unigram)
        items = sorted(self._unigram.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(w, (c + self.SMOOTHING) / total) for w, c in items]

    # -- Autocomplete ---------------------------------------------------------

    def autocomplete(self, prefix: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k (word, score) completions for a partial prefix.

        Searches the unigram vocabulary for words that start with prefix
        (case-insensitive) and ranks them by frequency.  Exact-match words
        are excluded because the user has already finished typing them.
        """
        prefix = prefix.lower()
        if not prefix:
            return []
        total = self._total_unigrams + self.SMOOTHING * len(self._unigram)
        matches = [
            (w, (c + self.SMOOTHING) / total)
            for w, c in self._unigram.items()
            if w.startswith(prefix) and w != prefix
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    # -- Federated merge (sum counts) -----------------------------------------

    def merge(self, other: "NgramModel") -> None:
        """Aggregate peer n-gram counts into the local model (federated sum)."""
        for w, c in other._unigram.items():
            self._unigram[w] += c
        self._total_unigrams += other._total_unigrams
        self._merge_node(self._root, other._root)

    def _merge_node(self, dst: TrieNode, src: TrieNode) -> None:
        for word, src_child in src.children.items():
            if word not in dst.children:
                dst.children[word] = TrieNode()
            dst_child = dst.children[word]
            dst_child.count += src_child.count
            dst.total += src_child.count
            self._merge_node(dst_child, src_child)

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a compact dict representation: context_key -> {word: count}."""
        flat: dict[str, dict[str, int]] = {}
        self._flatten(self._root, [], flat)
        return {
            "version": 1,
            "max_ngram": self.max_ngram,
            "unigram": dict(self._unigram),
            "total_unigrams": self._total_unigrams,
            "ngrams": flat,
        }

    def _flatten(self, node: TrieNode, context: list[str], out: dict) -> None:
        if context:
            ctx_key = " ".join(context)
            child_counts = {w: c.count for w, c in node.children.items() if c.count > 0}
            if child_counts:
                out[ctx_key] = child_counts
        for word, child in node.children.items():
            if len(context) < self.max_ngram - 1:
                self._flatten(child, context + [word], out)

    def from_dict(self, data: dict[str, Any]) -> None:
        """Load model state from a serialised dict."""
        self._unigram = defaultdict(int, data.get("unigram", {}))
        self._total_unigrams = int(data.get("total_unigrams", 0))
        self._root = TrieNode()
        for ctx_key, word_counts in data.get("ngrams", {}).items():
            ctx = ctx_key.split(" ")
            for word, count in word_counts.items():
                self._insert(ctx, word, int(count))

    def count_entries(self) -> int:
        """Return the approximate number of stored n-gram entries."""
        return sum(len(v) for v in self.to_dict().get("ngrams", {}).values())


# ---------------------------------------------------------------------------
# Seed model factory
# ---------------------------------------------------------------------------

def build_seed_model() -> NgramModel:
    """
    Return an NgramModel pre-seeded with generic English phrases.

    Sentences are loaded from Data/seed_corpus.txt at the project root.
    If that file is missing the model starts empty -- the application still
    runs, it just has no prior vocabulary until the user starts typing.
    """
    model = NgramModel()
    for sentence in _load_seed_texts():
        words = tokenize(sentence)
        if words:
            model.observe(words)
    return model
