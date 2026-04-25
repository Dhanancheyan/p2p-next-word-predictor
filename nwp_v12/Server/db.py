"""
db.py
SQLite WAL database layer and schema initialisation.

Initialisation order is critical:
  1. Create all core tables (CREATE TABLE IF NOT EXISTS) so migrations
     can safely reference them.
  2. Run db_migrations.run_migrations() to apply any pending schema changes.

SqliteDB is thread-safe: a single threading.Lock serialises all operations
on the shared connection.  WAL journal mode is enabled for better
read-write concurrency.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any


class SqliteDB:
    """Minimal thread-safe SQLite wrapper with WAL mode enabled."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a write statement and commit immediately."""
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Return the first row as a dict, or None if no rows match."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row else None

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Return all rows as a list of dicts."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def _create_core_tables(db: SqliteDB) -> None:
    """
    Create all core tables with IF NOT EXISTS guards.

    This must run before any migration because migrations read/write the
    config table and may reference other tables.
    """
    # config: key-value store for runtime settings and DB version.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        """
    )

    # models_registry: named model slots created by the user.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS models_registry (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            creation_type TEXT NOT NULL DEFAULT 'default'
        );
        """
    )

    # sessions: one row per typing session.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER,
            text_redacted TEXT NOT NULL,
            num_events INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_model_start"
        " ON sessions(model_id, start_ts);"
    )

    # events: per-session interaction events (accept, dismiss, etc.).
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_session_ts"
        " ON events(session_id, ts);"
    )

    # weights: metadata index for stored model weight files.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weights (
            weight_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            version INTEGER NOT NULL,
            path TEXT NOT NULL,
            arch_id TEXT NOT NULL,
            checksum TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_weights_model_kind_version"
        " ON weights(model_id, kind, version);"
    )


def init_db(db: SqliteDB) -> None:
    """
    Initialise the database: create tables then apply pending migrations.

    Safe to call on both fresh installs and existing databases.
    """
    _create_core_tables(db)
    from .db_migrations import run_migrations
    run_migrations(db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ts() -> int:
    """Return the current Unix timestamp as an integer."""
    return int(time.time())


def json_dumps(value: Any) -> str:
    """Serialise value to a compact, canonical JSON string."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_loads(value: str) -> Any:
    """Deserialise a JSON string."""
    return json.loads(value)
