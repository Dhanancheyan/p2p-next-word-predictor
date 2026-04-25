"""
model_registry.py
ModelSlot and ModelRegistry for hybrid LSTM + n-gram model lifecycle.

ModelSlot
---------
One registered model slot containing:
- HybridModelAgent  : LSTM + local/global n-gram engines.
- CacheAgent        : Session-level phrase cache for fast repeat predictions.
- PersonalizationLayer : Recency boost + user dictionary re-ranker.

Prediction pipeline (per slot):
  1. Get raw model predictions from HybridModelAgent.
  2. Check CacheAgent for phrase cache hits; blend at 0.25 weight.
  3. Apply recency score from CacheAgent at 0.10 weight.
  4. Re-rank via PersonalizationLayer (user dict + recency).
  5. Return top-k Suggestion objects.

Note: Suggestion.text has NO trailing space. The frontend acceptSuggestion
handler injects the space after insertion so backend results stay clean.

ModelRegistry
-------------
Lazy-loading registry: ModelSlot instances are created on first access and
cached for the lifetime of the server process.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .dl_module.cache_agent import CacheAgent
from .dl_module.hybrid_model import HybridModelAgent
from .dl_module.personalization import PersonalizationLayer


@dataclass
class Suggestion:
    text: str
    score: float
    source: str = "hybrid"
    mode: str = "next_word"
    partial_word: str = ""
    lstm_conf: float = 0.0


class ModelSlot:
    """One registered model: LSTM + local/global n-gram + cache + personalisation."""

    def __init__(self, model_id: str, data_dir: str) -> None:
        self.model_id = model_id
        self._data_dir = data_dir
        self.model_agent = HybridModelAgent()
        self.cache = CacheAgent()
        self.personalization = PersonalizationLayer()

        self._state_path = os.path.join(data_dir, f"hybrid_{model_id}.json")
        # Legacy path from pre-v11 installs (n-gram-only state file).
        self._legacy_state_path = os.path.join(data_dir, f"ngram_{model_id}.json")
        self._lstm_path = os.path.join(data_dir, f"lstm_{model_id}.json")
        self._pers_path = os.path.join(data_dir, f"pers_{model_id}.json")
        self._load()

    # -- Version proxies ------------------------------------------------------

    @property
    def local_version(self) -> int:
        return self.model_agent.local_version

    @local_version.setter
    def local_version(self, v: int) -> None:
        self.model_agent.local_version = v

    @property
    def global_version(self) -> int:
        return self.model_agent.global_version

    @property
    def weight_version(self) -> int:
        return self.model_agent.weight_version

    @property
    def engine(self) -> str:
        return self.model_agent.engine

    # -- Prediction -----------------------------------------------------------

    def predict(
        self,
        context_text: str,
        cursor_pos: int,
        k: int = 5,
        max_chars: int = 24,
    ) -> tuple[list[Suggestion], float]:
        """
        Return (suggestions, latency_ms).

        Blends model predictions with cache hits and personalisation scores.
        Filters out words longer than max_chars.
        """
        t0 = time.monotonic()

        prefix = context_text[:cursor_pos]
        words = [t.lower() for t in re.findall(r"[a-zA-Z']+|[0-9]+", prefix) if t]
        partial_word = ""
        if prefix and re.search(r"\w$", prefix):
            partial_word = words[-1] if words else ""

        # Cache lookup (skip for autocomplete -- cache stores full-word contexts).
        cache_context = words if not partial_word else words[:-1]
        cache_hits = self.cache.lookup(cache_context, k=k)

        # Model prediction.
        model_preds, _ = self.model_agent.predict(context_text, cursor_pos, k=k * 3)

        mode = (
            model_preds[0]["mode"] if model_preds
            else ("autocomplete" if partial_word else "next_word")
        )
        model_map = {p["text"].strip(): p["score"] for p in model_preds}
        source_map = {p["text"].strip(): p.get("source", "hybrid") for p in model_preds}
        lstm_conf_val = model_preds[0].get("lstm_conf", 0.0) if model_preds else 0.0
        # Cache bonus applies to next-word predictions only.
        cache_map = {w: s for w, s in cache_hits} if not partial_word else {}

        # Blend scores.
        all_words = set(model_map) | set(cache_map)
        blended: list[tuple[str, float]] = []
        for word in all_words:
            score = model_map.get(word, 0.0)
            score += cache_map.get(word, 0.0) * 0.25
            score += self.cache.recency_score(word) * 0.10
            blended.append((word, score))
        blended.sort(key=lambda x: x[1], reverse=True)

        # Personalisation re-ranking.
        candidates = [w for w, _ in blended[: k * 3]]
        base_scores = [s for _, s in blended[: k * 3]]
        reranked = self.personalization.rerank(candidates, base_scores)

        suggestions = [
            Suggestion(
                text=word,
                score=round(score, 6),
                source=source_map.get(
                    word, "cache" if word in cache_map else "hybrid"
                ),
                mode=mode,
                partial_word=partial_word,
                lstm_conf=lstm_conf_val,
            )
            for word, score in reranked
            if len(word) <= max_chars
        ][:k]

        latency_ms = (time.monotonic() - t0) * 1000
        return suggestions, round(latency_ms, 2)

    # -- Observation ----------------------------------------------------------

    def observe_text(self, text: str) -> None:
        """Update personalisation layer from typed text."""
        self.personalization.observe(text)

    def add_user_word(
        self, word: str, *, weight: float = 1.0, category: str = "custom"
    ) -> None:
        """Add a word to the user dictionary."""
        self.personalization.user_dict.add(word, weight=weight, category=category)

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """Persist n-gram state, LSTM weights, and personalisation data to disk."""
        os.makedirs(self._data_dir, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self.model_agent.to_persistence_dict(), f, ensure_ascii=False)
        self.model_agent.lstm.save(self._lstm_path)
        self.personalization.save(self._pers_path)

    def _load(self) -> None:
        """Load persisted state from disk, falling back gracefully on missing files."""
        # Prefer the current state path; fall back to legacy n-gram-only path.
        state_path = (
            self._state_path
            if os.path.exists(self._state_path)
            else self._legacy_state_path
        )
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    self.model_agent.from_persistence_dict(json.load(f))
            except Exception:
                pass
        self.model_agent.lstm.load(self._lstm_path)
        self.personalization.load(self._pers_path)

    # -- Federated helpers ----------------------------------------------------

    def get_delta_payload(self, top_k: int = 200) -> dict[str, Any]:
        """Return the hybrid sharing payload for peer gossip pulls."""
        return self.model_agent.get_federated_payload(top_k=top_k)

    def apply_peer_delta(self, delta_data: dict[str, Any]) -> None:
        """Merge a peer's federated payload into the global model."""
        self.model_agent.apply_federated_payload(delta_data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Lazy-loading registry of ModelSlot instances keyed by model_id."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._slots: dict[str, ModelSlot] = {}

    def get(self, model_id: str) -> ModelSlot:
        """Return the slot for model_id, creating it on first access."""
        if model_id not in self._slots:
            self._slots[model_id] = ModelSlot(model_id, self._data_dir)
        return self._slots[model_id]

    def save_all(self) -> None:
        """Persist all loaded model slots to disk (best-effort)."""
        for slot in self._slots.values():
            try:
                slot.save()
            except Exception:
                pass

    def ids(self) -> list[str]:
        """Return the list of currently loaded model IDs."""
        return list(self._slots.keys())
