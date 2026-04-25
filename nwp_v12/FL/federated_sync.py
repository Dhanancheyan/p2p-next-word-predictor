"""
federated_sync.py
Federated delta sharing: outbound push and inbound aggregation queue.

Responsibilities
----------------
- prepare_outbound : Serialise the local hybrid payload for sharing.
- share_delta      : Push payload to a list of peer nodes concurrently.
- receive_delta    : Enqueue an inbound peer payload for batch merge.
- flush_into_global: Drain the inbound queue and merge into the global model.
- fetch_peer_delta : Pull delta from a single peer (used by GossipEngine).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx


class FederatedSyncAgent:
    """Manages outbound hybrid payload sharing and inbound aggregation."""

    # Maximum n-gram entries per context key included in each delta.
    TOP_K = 200
    # Per-request HTTP timeout in seconds.
    TIMEOUT_S = 10.0

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        # Queue of inbound peer payloads waiting to be merged.
        self._pending_deltas: list[dict[str, Any]] = []
        self._last_share_ts: float = 0.0

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    def prepare_outbound(self, model_agent: Any, model_id: str) -> dict[str, Any]:
        """Build the federated payload envelope for sharing with peers."""
        return {
            "device_id": self.device_id,
            "model_id": model_id,
            "delta": model_agent.get_federated_payload(top_k=self.TOP_K),
            "ts": int(time.time()),
        }

    async def share_delta(
        self,
        payload: dict[str, Any],
        peer_urls: list[str],
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[dict[str, Any]]:
        """
        Push payload to all peer nodes concurrently.

        Returns a list of per-peer result dicts: {peer, ok, error?}.
        A semaphore can be supplied to cap the number of simultaneous
        outbound connections (server injects app.state.share_semaphore).
        """
        sem = semaphore or asyncio.Semaphore(10)

        async def _push(url: str) -> dict[str, Any]:
            async with sem:
                try:
                    async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                        r = await client.post(
                            f"{url.rstrip('/')}/federated/delta",
                            content=json.dumps(payload),
                            headers={"Content-Type": "application/json"},
                        )
                        r.raise_for_status()
                        return {"peer": url, "ok": True}
                except Exception as e:
                    return {"peer": url, "ok": False, "error": str(e)}

        results = list(
            await asyncio.gather(*[_push(u) for u in peer_urls], return_exceptions=False)
        )
        self._last_share_ts = time.time()
        return results

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    def receive_delta(self, payload: dict[str, Any]) -> None:
        """Enqueue an inbound peer payload for deferred merge by flush_into_global."""
        self._pending_deltas.append(payload)

    def flush_into_global(self, model_agent: Any) -> int:
        """
        Drain the inbound queue and merge each delta into the global model.

        Returns the number of deltas successfully merged.
        Note: the primary inbound path in app.py calls apply_peer_delta
        immediately on receipt for demo visibility; this queue handles the
        periodic background merge for any deltas enqueued via receive_delta.
        """
        if not self._pending_deltas:
            return 0
        deltas = self._pending_deltas[:]
        self._pending_deltas.clear()
        merged = 0
        for payload in deltas:
            delta_data = payload.get("delta", {})
            if not delta_data:
                continue
            if hasattr(model_agent, "apply_federated_payload"):
                model_agent.apply_federated_payload(delta_data)
                merged += 1
        return merged

    async def fetch_peer_delta(self, peer_url: str, model_id: str) -> dict[str, Any] | None:
        """Pull the current delta payload from a single peer (used by GossipEngine)."""
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                r = await client.get(
                    f"{peer_url.rstrip('/')}/federated/weights",
                    params={"model_id": model_id},
                )
                r.raise_for_status()
                return r.json()
        except Exception:
            return None

    def stats(self) -> dict[str, Any]:
        """Return a summary of sync agent state for diagnostics."""
        return {
            "pending_deltas": len(self._pending_deltas),
            "last_share_ts": int(self._last_share_ts),
            "top_k": self.TOP_K,
        }
