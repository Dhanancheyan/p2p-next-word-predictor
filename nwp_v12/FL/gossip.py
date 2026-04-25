"""
gossip.py
Decentralised gossip federated learning engine with peer reputation scoring.

Protocol
--------
Each node periodically:
  1. Picks the highest-reputation available peer.
  2. Fetches their hybrid delta via GET /federated/weights.
  3. Merges: n-gram counts are summed; LSTM weights are L2-normalised and blended.
  4. Updates the peer's reputation score based on latency and outcome.

Reputation scoring uses exponential moving average latency plus a simple
success/failure counter so unreliable peers are naturally deprioritised.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Peer Reputation
# ---------------------------------------------------------------------------

@dataclass
class PeerRecord:
    url: str
    peer_id: str = ""
    score: float = 1.0
    successes: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0
    last_seen: float = field(default_factory=time.time)
    model_version: int = 0


class PeerReputation:
    """
    Tracks per-peer reliability and latency scores.

    Scores are bounded to [0.0, 2.0].  A fresh peer starts at 1.0.
    Each successful sync boosts the score by SUCCESS_BOOST minus a
    latency penalty; each failure reduces by FAILURE_PENALTY.
    """

    LATENCY_THRESHOLD_MS = 500.0
    SUCCESS_BOOST = 0.1
    FAILURE_PENALTY = 0.2
    LATENCY_COEFF = 0.01

    def __init__(self) -> None:
        self._peers: dict[str, PeerRecord] = {}

    def upsert(self, url: str, peer_id: str = "", model_version: int = 0) -> PeerRecord:
        """Insert a new peer record or refresh metadata for an existing one."""
        if url not in self._peers:
            self._peers[url] = PeerRecord(
                url=url, peer_id=peer_id, model_version=model_version
            )
        else:
            r = self._peers[url]
            if peer_id:
                r.peer_id = peer_id
            r.model_version = model_version
        return self._peers[url]

    def record_success(self, url: str, latency_ms: float) -> None:
        """Update EMA latency and boost score after a successful sync."""
        r = self._peers.get(url)
        if r is None:
            return
        r.successes += 1
        r.last_seen = time.time()
        # Exponential moving average: 80% old, 20% new sample.
        r.avg_latency_ms = (
            latency_ms
            if r.avg_latency_ms == 0
            else 0.8 * r.avg_latency_ms + 0.2 * latency_ms
        )
        lat_penalty = (
            max(0.0, (r.avg_latency_ms - self.LATENCY_THRESHOLD_MS) / 100.0)
            * self.LATENCY_COEFF
        )
        r.score = min(2.0, r.score + self.SUCCESS_BOOST - lat_penalty)

    def record_failure(self, url: str) -> None:
        """Penalise a peer after a failed sync attempt."""
        r = self._peers.get(url)
        if r is None:
            return
        r.failures += 1
        r.score = max(0.0, r.score - self.FAILURE_PENALTY)

    def sorted_peers(self, min_score: float = 0.1) -> list[PeerRecord]:
        """Return peers with score >= min_score, sorted highest first."""
        return sorted(
            [p for p in self._peers.values() if p.score >= min_score],
            key=lambda p: p.score,
            reverse=True,
        )

    def all_peer_dicts(self) -> list[dict[str, Any]]:
        """Serialise all peer records for the /local/peers/reputation endpoint."""
        return [
            {
                "url": p.url,
                "peer_id": p.peer_id,
                "score": round(p.score, 3),
                "successes": p.successes,
                "failures": p.failures,
                "avg_latency_ms": round(p.avg_latency_ms, 1),
                "last_seen": int(p.last_seen),
                "model_version": p.model_version,
            }
            for p in self._peers.values()
        ]


# ---------------------------------------------------------------------------
# Gossip Engine
# ---------------------------------------------------------------------------

class GossipEngine:
    """
    Runs a single gossip round: fetch one peer's delta and merge it locally.

    Peers are tried in reputation order (highest score first), with
    unranked peers shuffled after. The round returns on the first
    successful merge or exhausts all candidates.
    """

    def __init__(
        self,
        *,
        own_url: str,
        reputation: PeerReputation | None = None,
    ) -> None:
        self.own_url = own_url.rstrip("/")
        self.reputation = reputation or PeerReputation()

    async def run_round(
        self,
        *,
        model_id: str,
        peer_urls: list[str],
        get_slot_fn: Any,
        timeout_s: float = 15.0,
    ) -> dict[str, Any]:
        """
        Execute one gossip round for the given model.

        Parameters
        ----------
        model_id    : ID of the model slot to update.
        peer_urls   : Full list of candidate peer base URLs.
        get_slot_fn : Callable (model_id) -> ModelSlot from the model registry.
        timeout_s   : Per-peer HTTP timeout.

        Returns a result dict with 'ok', 'round_id', and either peer info
        or a 'reason' string on failure.
        """
        import random

        round_id = str(uuid.uuid4())[:12]

        # Exclude ourselves from the candidate list.
        candidates = [u for u in peer_urls if u.rstrip("/") != self.own_url]
        if not candidates:
            return {"ok": False, "reason": "no_peers", "round_id": round_id}

        # Order by reputation score, append unranked peers in random order.
        rep_sorted = self.reputation.sorted_peers(min_score=0.05)
        rep_urls = [r.url for r in rep_sorted if r.url in candidates]
        remaining = [u for u in candidates if u not in rep_urls]
        random.shuffle(remaining)
        ordered = rep_urls + remaining

        for peer_url in ordered:
            t0 = time.monotonic()
            try:
                delta_data = await self._fetch_delta(
                    peer_url, model_id=model_id, timeout_s=timeout_s
                )
                if delta_data is None:
                    self.reputation.record_failure(peer_url)
                    continue

                slot = get_slot_fn(model_id)
                slot.apply_peer_delta(delta_data)

                latency_ms = (time.monotonic() - t0) * 1000
                self.reputation.upsert(peer_url)
                self.reputation.record_success(peer_url, latency_ms)

                return {
                    "ok": True,
                    "round_id": round_id,
                    "peer_url": peer_url,
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception:
                self.reputation.record_failure(peer_url)
                continue

        return {"ok": False, "reason": "all_peers_failed", "round_id": round_id}

    async def _fetch_delta(
        self,
        peer_url: str,
        *,
        model_id: str,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        """Fetch the federated weights payload from a single peer."""
        url = f"{peer_url.rstrip('/')}/federated/weights"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.get(url, params={"model_id": model_id})
                r.raise_for_status()
                return r.json()
        except Exception:
            return None
