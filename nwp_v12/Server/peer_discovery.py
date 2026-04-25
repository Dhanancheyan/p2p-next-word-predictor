"""
peer_discovery.py
Automatic NWP instance discovery within the port range 8001-8020.

PeerDiscovery scans localhost (and optionally the LAN /24 subnet) by probing
GET /health on each candidate URL concurrently.  Only instances that return
{"ok": true} are reported as discovered peers.

Port range constraint
---------------------
All discovery is strictly limited to ports 8001-8020.  This prevents accidental
connections to unrelated services and keeps the scan fast (20 ports max).

LAN scanning
------------
LAN scanning is disabled by default and must be explicitly enabled via the
discovery_enable_lan config key (or the FL Settings toggle in the web UI).
When enabled, all hosts on the local /24 subnet are probed on ports 8001-8020.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Port range enforcement
# ---------------------------------------------------------------------------

PORT_MIN = 8001
PORT_MAX = 8020


def _validate_port_range(ports: list[int]) -> list[int]:
    """Return only ports within the allowed 8001-8020 range."""
    return [p for p in ports if PORT_MIN <= p <= PORT_MAX]


# ---------------------------------------------------------------------------
# Peer record
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredPeer:
    url: str
    peer_id: str
    latency_ms: float
    status: str          # "ok" or "error"
    meta: dict[str, Any] = field(default_factory=dict)
    last_seen: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "peer_id": self.peer_id,
            "latency_ms": round(self.latency_ms, 1),
            "status": self.status,
            "meta": self.meta,
            "last_seen": self.last_seen,
        }


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

async def _probe(url: str, *, timeout_s: float = 1.5) -> DiscoveredPeer | None:
    """
    Probe a single URL via GET /health.

    Returns a DiscoveredPeer on success, None on any error or non-ok response.
    """
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{url}/health")
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                return None
            latency_ms = (time.monotonic() - t0) * 1000
            peer_id = str(data.get("peer_id", url))
            return DiscoveredPeer(
                url=url,
                peer_id=peer_id,
                latency_ms=latency_ms,
                status="ok",
                meta={k: v for k, v in data.items() if k not in ("ok", "peer_id")},
            )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Local IP detection
# ---------------------------------------------------------------------------

def _local_ipv4_addresses() -> list[str]:
    """Return non-loopback local IPv4 addresses for LAN subnet scanning."""
    addrs: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.append(ip)
    except Exception:
        pass
    if not addrs:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                addrs.append(s.getsockname()[0])
        except Exception:
            pass
    return list(set(addrs))


# ---------------------------------------------------------------------------
# Peer Discovery
# ---------------------------------------------------------------------------

class PeerDiscovery:
    """
    Manages background peer scanning and caches the most recent results.

    Parameters
    ----------
    own_url       : This instance's base URL (excluded from scan results).
    port_range    : Ports to probe (default 8001-8020, enforced strictly).
    probe_timeout_s : Per-probe HTTP timeout in seconds.
    """

    DEFAULT_PORTS = list(range(PORT_MIN, PORT_MAX + 1))

    def __init__(
        self,
        *,
        own_url: str = "",
        port_range: list[int] | None = None,
        probe_timeout_s: float = 1.5,
    ) -> None:
        self.own_url = own_url.rstrip("/")
        self.port_range = _validate_port_range(port_range or self.DEFAULT_PORTS)
        self.probe_timeout_s = probe_timeout_s
        self._peers: dict[str, DiscoveredPeer] = {}
        self._lock = asyncio.Lock()
        self._last_scan_ts: int = 0
        self._lan_enabled: bool = False

    @property
    def peers(self) -> list[DiscoveredPeer]:
        return list(self._peers.values())

    @property
    def last_scan_ts(self) -> int:
        return self._last_scan_ts

    @property
    def lan_enabled(self) -> bool:
        return self._lan_enabled

    async def scan(self, *, enable_lan: bool = False) -> list[DiscoveredPeer]:
        """
        Probe all candidate URLs concurrently and cache the results.

        Parameters
        ----------
        enable_lan : When True, also probe all hosts on the local /24 subnet.
        """
        self._lan_enabled = enable_lan
        candidate_urls = [
            u for u in self._build_candidate_urls(enable_lan=enable_lan)
            if u.rstrip("/") != self.own_url
        ]

        results = await asyncio.gather(
            *[_probe(u, timeout_s=self.probe_timeout_s) for u in candidate_urls],
            return_exceptions=True,
        )

        found: dict[str, DiscoveredPeer] = {}
        for result in results:
            if isinstance(result, DiscoveredPeer):
                found[result.url] = result

        async with self._lock:
            self._peers = found
            self._last_scan_ts = int(time.time())

        return list(found.values())

    def get_active_urls(self) -> list[str]:
        """Return URLs of all currently discovered peers with status 'ok'."""
        return [p.url for p in self._peers.values() if p.status == "ok"]

    def _build_candidate_urls(self, *, enable_lan: bool) -> list[str]:
        urls: list[str] = []

        # Always scan localhost on the allowed port range.
        for port in self.port_range:
            urls.append(f"http://127.0.0.1:{port}")

        # Optionally scan all hosts on the local /24 subnet (same port range).
        if enable_lan:
            for local_ip in _local_ipv4_addresses():
                try:
                    net = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
                    for host in net.hosts():
                        hip = str(host)
                        if hip == local_ip:
                            continue
                        for port in self.port_range:
                            urls.append(f"http://{hip}:{port}")
                except Exception:
                    continue

        return list(dict.fromkeys(urls))  # deduplicate while preserving order
