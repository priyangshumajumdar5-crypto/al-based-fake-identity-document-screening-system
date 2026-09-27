"""
Database connection management for SQLite.
Uses stdlib sqlite3 — no ORM required for this prototype.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file."""
    return settings.database_path_absolute


def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection.

    Row factory is set to sqlite3.Row so columns are accessible by name.

    Returns:
        An open sqlite3.Connection.
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db_session() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a connection and commits/rolls back.

    Usage::

        with db_session() as conn:
            conn.execute(...)
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database() -> None:
    """
    Create all tables if they do not already exist and run backward-compatible migrations.
    Safe to call multiple times (idempotent).
    """
    from app.database.models import CREATE_TABLES_SQL

    with db_session() as conn:
        for statement in CREATE_TABLES_SQL:
            conn.execute(statement)

        # Backward-compatible migration: ensure passport_photo_path exists in passport_records
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(passport_records);").fetchall()]
            if "passport_photo_path" not in cols:
                conn.execute("ALTER TABLE passport_records ADD COLUMN passport_photo_path TEXT;")
                logger.info("Migrated passport_records table: added passport_photo_path column.")
        except Exception as exc:
            logger.warning("Migration check on passport_records failed: %s", exc)

    logger.info("Database initialised at %s", get_db_path())
