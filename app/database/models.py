"""
Database table DDL and query constants.
All SQL lives here — not scattered across the application.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Table: passport_records
# Contains ONLY synthetic/test records for the Phase 1 prototype.
# NOT connected to any government database.
# ─────────────────────────────────────────────────────────────────────────────

CREATE_PASSPORT_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS passport_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    passport_number     TEXT    NOT NULL UNIQUE,
    name                TEXT    NOT NULL,
    date_of_birth       TEXT    NOT NULL,   -- YYYY-MM-DD
    place_of_birth      TEXT,
    place_of_issue      TEXT,
    date_of_issue       TEXT    NOT NULL,   -- YYYY-MM-DD
    date_of_expiry      TEXT    NOT NULL,   -- YYYY-MM-DD
    sex                 TEXT    NOT NULL,   -- M / F / X
    country_code        TEXT    NOT NULL DEFAULT 'IND',
    passport_type       TEXT    NOT NULL DEFAULT 'P',
    visa_number         TEXT,
    visa_validity       TEXT,               -- YYYY-MM-DD
    status              TEXT    NOT NULL DEFAULT 'VALID',
    passport_photo_path TEXT,
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_SCREENING_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS screening_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    passport_number     TEXT,
    ocr_success         INTEGER NOT NULL DEFAULT 0,  -- 0/1
    validation_status   TEXT,
    database_match      TEXT,
    processed_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    notes               TEXT
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Table: passports (real CSV-seeded records — Module 2 v2)
# Created by seed_from_csv.py but also auto-created here for safety.
# ─────────────────────────────────────────────────────────────────────────────

CREATE_PASSPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS passports (
    passport_number     TEXT PRIMARY KEY,
    surname             TEXT,
    given_name          TEXT,
    date_of_birth       TEXT,
    place_of_birth      TEXT,
    date_of_issue       TEXT,
    place_of_issue      TEXT,
    date_of_expiry      TEXT,
    gender              TEXT,
    passport_photo_path TEXT
);
"""

CREATE_TABLES_SQL = [
    CREATE_PASSPORT_RECORDS_TABLE,
    CREATE_SCREENING_LOG_TABLE,
    CREATE_PASSPORTS_TABLE,
]

# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

SELECT_BY_PASSPORT_NUMBER = """
SELECT * FROM passport_records WHERE passport_number = ?;
"""

INSERT_PASSPORT_RECORD = """
INSERT OR IGNORE INTO passport_records
    (passport_number, name, date_of_birth, place_of_birth, place_of_issue,
     date_of_issue, date_of_expiry, sex, country_code, passport_type,
     visa_number, visa_validity, status, passport_photo_path, notes)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

INSERT_SCREENING_LOG = """
INSERT INTO screening_log
    (passport_number, ocr_success, validation_status, database_match, notes)
VALUES (?, ?, ?, ?, ?);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Queries for `passports` table (real CSV-seeded records)
# ─────────────────────────────────────────────────────────────────────────────

SELECT_FROM_PASSPORTS = """
SELECT * FROM passports WHERE passport_number = ?;
"""

SELECT_ALL_PASSPORT_NUMBERS = """
SELECT passport_number FROM passports;
"""
