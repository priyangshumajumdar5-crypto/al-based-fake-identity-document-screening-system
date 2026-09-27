"""
Seed SQLite database at data/database/passport_screening.db table `passports` from CSV file.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


def seed_from_csv(
    csv_path: str = "data/draft_passport_database.csv",
    images_dir: str = "passport",
    db_path: str = "data/database/passport_screening.db",
) -> int:
    """
    Load a CSV into SQLite database at `db_path`, table `passports`, replacing any existing data.

    Args:
        csv_path: Path to the input CSV file.
        images_dir: Base directory containing passport images.
        db_path: Path to the SQLite database file.

    Returns:
        Number of rows inserted.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: CSV file '{csv_path}' not found.")
        return 0

    sqlite_db = Path(db_path)
    sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(sqlite_db))
    cursor = conn.cursor()

    # Drop existing table if present and recreate
    cursor.execute("DROP TABLE IF EXISTS passports;")

    cursor.execute("""
        CREATE TABLE passports (
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
    """)

    # Also maintain passport_records table for system compatibility
    cursor.execute("DROP TABLE IF EXISTS passport_records;")
    cursor.execute("""
        CREATE TABLE passport_records (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            passport_number     TEXT UNIQUE,
            name                TEXT,
            surname             TEXT,
            given_name          TEXT,
            date_of_birth       TEXT,
            place_of_birth      TEXT,
            place_of_issue      TEXT,
            date_of_issue       TEXT,
            date_of_expiry      TEXT,
            sex                 TEXT,
            gender              TEXT,
            country_code        TEXT DEFAULT 'IND',
            passport_type       TEXT DEFAULT 'P',
            status              TEXT DEFAULT 'VALID',
            passport_photo_path TEXT
        );
    """)

    inserted = 0

    def _clean(val: str | None) -> str:
        if not val or val.strip().lower() == "null":
            return ""
        return val.strip()

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = _clean(row.get("image_file_name") or row.get("image_filename"))
            pn = _clean(row.get("passport_number"))
            surname = _clean(row.get("surname"))
            given_name = _clean(row.get("given_name"))
            dob = _clean(row.get("date_of_birth"))
            pob = _clean(row.get("place_of_birth"))
            doi = _clean(row.get("date_of_issue"))
            poi = _clean(row.get("place_of_issue"))
            doe = _clean(row.get("date_of_expiry"))
            gender = _clean(row.get("gender") or row.get("sex"))

            photo_path = f"{images_dir}/{img_name}" if img_name else ""

            if not pn:
                continue

            # Insert into `passports` table
            cursor.execute("""
                INSERT OR REPLACE INTO passports (
                    passport_number, surname, given_name, date_of_birth,
                    place_of_birth, date_of_issue, place_of_issue, date_of_expiry,
                    gender, passport_photo_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (pn, surname, given_name, dob, pob, doi, poi, doe, gender, photo_path))

            # Insert into `passport_records` table
            full_name = f"{surname} {given_name}".strip()
            cursor.execute("""
                INSERT OR REPLACE INTO passport_records (
                    passport_number, name, surname, given_name, date_of_birth,
                    place_of_birth, place_of_issue, date_of_issue, date_of_expiry,
                    sex, gender, country_code, passport_type, status, passport_photo_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IND', 'P', 'VALID', ?);
            """, (pn, full_name, surname, given_name, dob, pob, poi, doi, doe, gender, gender, photo_path))

            inserted += 1

    conn.commit()
    conn.close()

    print(f"Successfully loaded and inserted {inserted} row(s) into '{db_path}' (table: 'passports').")
    return inserted


if __name__ == "__main__":
    seed_from_csv()
