"""
ws.py
WebSocket broadcast hub for server-to-client event notifications.

All connected clients receive every broadcast.  Disconnected clients are
removed automatically when a send fails.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class WebSocketHub:
    """Thread-safe WebSocket connection manager with JSON broadcast support."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection from the active set."""
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        """
        Send payload as JSON to all connected clients.

        Clients that raise an exception during send are disconnected silently.
        """
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)
