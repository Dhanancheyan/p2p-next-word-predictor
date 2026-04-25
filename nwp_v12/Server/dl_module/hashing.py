"""
hashing.py
Deterministic hashing utilities for text and JSON payloads.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_dumps(value: Any) -> str:
    """Serialise value to a canonical, sort-keyed JSON string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex_of_canonical_json(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON representation of value."""
    return sha256_hex(canonical_json_dumps(value).encode("utf-8"))
