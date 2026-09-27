"""
Pytest tests for database module.
Uses a temporary database file to keep tests isolated.
"""
import pytest
import sqlite3
import os
from pathlib import Path


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """Redirect database to a temp path for every test."""
    db_file = str(tmp_path / "test_passport.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    # Force settings to re-read env
    from app.core import config as cfg
    cfg.settings.DATABASE_PATH = db_file
    yield db_file
    # Cleanup handled by tmp_path


class TestDatabaseConnection:
    def test_get_connection_returns_connection(self, use_temp_db):
        from app.database.database import initialize_database, get_connection
        initialize_database()
        conn = get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_row_factory_set(self, use_temp_db):
        from app.database.database import initialize_database, get_connection
        initialize_database()
        conn = get_connection()
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_db_session_commits(self, use_temp_db):
        from app.database.database import initialize_database, db_session
        initialize_database()

        with db_session() as conn:
            conn.execute(
                "INSERT INTO passport_records "
                "(passport_number, name, date_of_birth, date_of_issue, "
                "date_of_expiry, sex) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("T1234567", "TEST USER", "1990-01-01", "2020-01-01", "2030-01-01", "M")
            )
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM passport_records WHERE passport_number = ?",
                ("T1234567",)
            ).fetchone()
        assert row is not None
        assert row["name"] == "TEST USER"

    def test_db_session_rollback_on_error(self, use_temp_db):
        from app.database.database import initialize_database, db_session
        initialize_database()
        with pytest.raises(Exception):
            with db_session() as conn:
                conn.execute("SELECT * FROM nonexistent_table_xyz")


class TestDatabaseInitialisation:
    def test_tables_created(self, use_temp_db):
        from app.database.database import initialize_database, db_session
        initialize_database()
        with db_session() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "passport_records" in table_names
        assert "screening_log" in table_names

    def test_init_idempotent(self, use_temp_db):
        from app.database.database import initialize_database
        initialize_database()
        initialize_database()  # Second call should not raise

    def test_passport_records_columns(self, use_temp_db):
        from app.database.database import initialize_database, db_session
        initialize_database()
        with db_session() as conn:
            cols = conn.execute("PRAGMA table_info(passport_records)").fetchall()
        col_names = [c["name"] for c in cols]
        for required in ["id", "passport_number", "name", "date_of_birth",
                         "date_of_expiry", "status"]:
            assert required in col_names


class TestDatabaseSeed:
    def test_seed_inserts_records(self, use_temp_db):
        from app.database.database import initialize_database, db_session
        from app.database.seed import seed_database, SYNTHETIC_RECORDS
        seed_database()
        with db_session() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM passport_records"
            ).fetchone()[0]
        assert count > 0

    def test_seed_idempotent(self, use_temp_db):
        from app.database.seed import seed_database, SYNTHETIC_RECORDS
        seed_database()
        seed_database()
        from app.database.database import db_session
        with db_session() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM passport_records"
            ).fetchone()[0]
        assert count == len(SYNTHETIC_RECORDS)

    def test_seed_valid_record_exists(self, use_temp_db):
        from app.database.seed import seed_database
        from app.database.database import db_session
        seed_database()
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM passport_records WHERE passport_number = ?",
                ("A1234567",)
            ).fetchone()
        assert row is not None
        assert row["status"] == "VALID"

    def test_seed_expired_record_exists(self, use_temp_db):
        from app.database.seed import seed_database
        from app.database.database import db_session
        seed_database()
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM passport_records WHERE passport_number = ?",
                ("F6789012",)
            ).fetchone()
        assert row is not None
        assert row["status"] == "EXPIRED"
