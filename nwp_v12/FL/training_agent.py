"""
training_agent.py
Real-time, non-blocking local model update pipeline.

Triggered after a session ends:
  1. Updates the local n-gram model and runs a small LSTM fine-tune (instant, <1 ms).
  2. Updates the session-level phrase cache.
  3. Flushes any queued inbound peer deltas into the global model.
  4. Optionally fires a federated delta push to peers in the background.

All heavy network work runs as an asyncio background task so prediction
latency is never blocked by training or sharing.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z']+|[0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t]


class TrainingAgent:
    """
    Coordinates real-time local learning after each session.

    Parameters
    ----------
    model_agent          : HybridModelAgent instance to train.
    cache_agent          : CacheAgent instance to update.
    federated_sync_agent : FederatedSyncAgent for outbound sharing.
    hub_broadcast_fn     : Optional async callable to push WebSocket events.
    """

    def __init__(
        self,
        model_agent: Any,
        cache_agent: Any,
        federated_sync_agent: Any,
        hub_broadcast_fn: Callable[[dict], Coroutine] | None = None,
    ) -> None:
        self.model = model_agent
        self.cache = cache_agent
        self.sync = federated_sync_agent
        self._broadcast = hub_broadcast_fn
        self._share_semaphore = asyncio.Semaphore(10)

    async def on_sentence_complete(
        self,
        text: str,
        model_id: str,
        peer_urls: list[str],
        auto_share: bool = False,
    ) -> None:
        """
        Called when the user ends a session.

        Steps (in order):
          1. Broadcast 'training_started' so the UI indicator activates immediately.
          2. Observe text in the local n-gram and run a short LSTM fine-tune.
          3. Update the session cache.
          4. Flush queued inbound peer deltas into the global model.
          5. Broadcast 'training_complete'.
          6. If auto_share is True, push delta to peers in a background task.
        """
        words = _tokenize(text)
        if not words:
            return

        if self._broadcast:
            await self._broadcast({"type": "training_started", "model_id": model_id})

        # Step 2: local n-gram + LSTM update (synchronous, sub-millisecond for n-gram).
        self.model.observe(text)
        logger.debug(
            "TrainingAgent: observed %d words, local_version=%d",
            len(words),
            self.model.local_version,
        )

        # Step 3: session cache update.
        self.cache.observe(text)

        # Step 4: merge any pending inbound deltas from other peers.
        merged = self.sync.flush_into_global(self.model)
        if merged > 0:
            logger.debug(
                "TrainingAgent: flushed %d peer deltas into global model", merged
            )

        if self._broadcast:
            await self._broadcast({
                "type": "training_complete",
                "model_id": model_id,
                "local_version": self.model.local_version,
                "global_version": self.model.global_version,
                "words_learned": len(words),
                "engine": getattr(self.model, "engine", "hybrid"),
            })

        # Step 6: non-blocking federated push.
        if auto_share and peer_urls:
            asyncio.create_task(self._share_in_background(model_id, peer_urls))

    async def _share_in_background(
        self,
        model_id: str,
        peer_urls: list[str],
    ) -> None:
        """Push delta to peers as a background task; never blocks predictions."""
        try:
            payload = self.sync.prepare_outbound(self.model, model_id)
            results = await self.sync.share_delta(
                payload, peer_urls, semaphore=self._share_semaphore
            )
            ok_count = sum(1 for r in results if r.get("ok"))
            logger.debug(
                "TrainingAgent: shared delta to %d/%d peers", ok_count, len(peer_urls)
            )
            if self._broadcast:
                await self._broadcast({
                    "type": "share_complete",
                    "model_id": model_id,
                    "peers_ok": ok_count,
                    "peers_total": len(results),
                })
        except Exception as e:
            logger.warning("TrainingAgent: background share failed: %s", e)

    def manual_train(self, texts: list[str]) -> int:
        """
        Synchronously train on a list of historic session texts.

        Calls train_texts on the model agent (which runs both n-gram observe
        and LSTM fine-tune), then updates the session cache for each text.
        Returns the total number of words learned.
        """
        total = (
            self.model.train_texts(texts)
            if hasattr(self.model, "train_texts")
            else 0
        )
        for text in texts:
            if _tokenize(text):
                self.cache.observe(text)
        # Fallback: if train_texts returned 0 (no LSTM), count via observe.
        if not total:
            for text in texts:
                words = _tokenize(text)
                if words:
                    self.model.observe(text)
                    total += len(words)
        return total
