"""
db_migrations.py
Version-guarded SQLite schema migration system.

Protocol
--------
- Each migration is identified by a monotonically increasing version integer.
- The current DB version is stored in the config table as "db_version".
- Fresh installs start at version 0 and run all migrations in sequence.
- Existing installs detect their current version and skip already-applied ones.
- All migration functions use IF NOT EXISTS / existence checks so they are
  safe to run on both fresh and existing databases.
- Failures are logged but do not crash startup (best-effort).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import SqliteDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _table_exists(db: "SqliteDB", table_name: str) -> bool:
    """Return True if the named table exists in the database."""
    row = db.query_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    )
    return row is not None


def _get_db_version(db: "SqliteDB") -> int:
    """Return the current migration version (0 for a fresh database)."""
    if not _table_exists(db, "config"):
        return 0
    row = db.query_one("SELECT value_json FROM config WHERE key='db_version';")
    if not row:
        return 0
    try:
        return int(json.loads(row["value_json"]))
    except Exception:
        return 0


def _set_db_version(db: "SqliteDB", version: int) -> None:
    db.execute(
        "INSERT INTO config(key, value_json) VALUES('db_version', ?)"
        " ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json;",
        (json.dumps(version),),
    )


# ---------------------------------------------------------------------------
# Migrations (one function per version, in order)
# ---------------------------------------------------------------------------

def _migration_1(db: "SqliteDB") -> None:
    """Remove legacy preset profile tables and rows from pre-v7 databases."""
    if _table_exists(db, "profiles"):
        db.execute("DROP TABLE IF EXISTS profiles;")
        logger.info("db_migration 1: dropped legacy 'profiles' table")

    if _table_exists(db, "models_registry"):
        for old_id in ("programming", "casual", "academic"):
            db.execute(
                "DELETE FROM models_registry WHERE model_id=? AND creation_type='default';",
                (old_id,),
            )
        logger.info("db_migration 1: removed legacy preset rows from models_registry")


def _migration_2(db: "SqliteDB") -> None:
    """Add peer_reputation table for gossip FL scoring."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS peer_reputation (
            url TEXT PRIMARY KEY,
            peer_id TEXT NOT NULL DEFAULT '',
            score REAL NOT NULL DEFAULT 1.0,
            successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms REAL NOT NULL DEFAULT 0.0,
            last_seen INTEGER NOT NULL DEFAULT 0,
            model_version INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    logger.info("db_migration 2: created peer_reputation table")


def _migration_3(db: "SqliteDB") -> None:
    """Add personalization_state table for per-model personalisation snapshots."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS personalization_state (
            model_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    logger.info("db_migration 3: created personalization_state table")


def _migration_4(db: "SqliteDB") -> None:
    """Add ngram_sync_log table to audit federated delta exchanges."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS ngram_sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'inbound',
            peer_url TEXT NOT NULL DEFAULT '',
            ts INTEGER NOT NULL DEFAULT 0,
            entries_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    logger.info("db_migration 4: created ngram_sync_log table")


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, object]] = [
    (1, _migration_1),
    (2, _migration_2),
    (3, _migration_3),
    (4, _migration_4),
]

LATEST_VERSION: int = max(v for v, _ in _MIGRATIONS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_migrations(db: "SqliteDB") -> None:
    """
    Apply all pending migrations in ascending version order.

    Safe on fresh installs and upgrades.  Individual migration failures are
    logged but do not abort the startup sequence.
    """
    current = _get_db_version(db)
    logger.info("Database version: %d  (latest: %d)", current, LATEST_VERSION)

    for version, migration_fn in sorted(_MIGRATIONS, key=lambda x: x[0]):
        if version > current:
            logger.info("Applying migration %d ...", version)
            try:
                migration_fn(db)
                _set_db_version(db, version)
                logger.info("Migration %d complete", version)
            except Exception as exc:
                logger.error("Migration %d failed: %s", version, exc)
