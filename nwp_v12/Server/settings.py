"""
settings.py
Application settings dataclass and device-ID persistence.

TrainerSettings is an immutable dataclass constructed once at startup by
build_settings().  The device ID is generated on first run and saved to
data_dir/device_id.txt so it survives restarts.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerSettings:
    """Immutable runtime configuration for the NWP server."""

    data_dir: str    # root directory for all runtime data (DB, model files, etc.)
    device_id: str   # stable per-device identifier, persisted in data_dir
    port: int = 8001 # HTTP port (must be in 8001-8020)

    @property
    def db_path(self) -> str:
        """Absolute path to the SQLite database file."""
        return os.path.join(self.data_dir, "trainer.sqlite3")

    @property
    def own_url(self) -> str:
        """Base URL used for self-exclusion during peer discovery."""
        return f"http://127.0.0.1:{self.port}"


def _load_or_create_device_id(data_dir: str) -> str:
    """
    Return the persisted device ID, creating one if it does not exist.

    The ID is stored as plain text in data_dir/device_id.txt.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "device_id.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            device_id = f.read().strip()
        if device_id:
            return device_id
    device_id = f"device-{uuid.uuid4()}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(device_id)
    return device_id


def build_settings(*, data_dir: str, port: int = 8001) -> TrainerSettings:
    """
    Construct TrainerSettings from the given data directory and port.

    Loads (or generates) the device ID from data_dir/device_id.txt.
    """
    return TrainerSettings(
        data_dir=data_dir,
        device_id=_load_or_create_device_id(data_dir),
        port=port,
    )
