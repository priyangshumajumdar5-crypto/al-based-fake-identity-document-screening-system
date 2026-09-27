"""
Database seeding module — inserts synthetic test records.
ALL data here is fictional and for prototype demonstration only.
No real passport numbers, names, or identity documents are used.

Safe to call multiple times (INSERT OR IGNORE prevents duplicates).
"""
from __future__ import annotations

from typing import List, Tuple

from app.core.logging_config import get_logger
from app.database.database import db_session, initialize_database
from app.database.models import INSERT_PASSPORT_RECORD

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TEST RECORDS
# Fields: passport_number, name, date_of_birth, place_of_birth,
#         place_of_issue, date_of_issue, date_of_expiry, sex,
#         country_code, passport_type, visa_number, visa_validity,
#         status, passport_photo_path, notes
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_RECORDS: List[Tuple] = [
    # ── VALID records ────────────────────────────────────────────────────────
    (
        "A1234567", "ARJUN KUMAR SHARMA", "1990-05-15",
        "NEW DELHI", "NEW DELHI", "2019-03-20", "2029-03-19",
        "M", "IND", "P", "BV2024001", "2025-12-31",
        "VALID", "data/samples/portraits/A1234567.jpg", "Synthetic test record — VALID"
    ),
    (
        "B2345678", "PRIYA MEENAKSHI IYER", "1985-11-22",
        "CHENNAI", "CHENNAI", "2021-06-10", "2031-06-09",
        "F", "IND", "P", "BV2024002", "2026-06-30",
        "VALID", "data/samples/portraits/B2345678.jpg", "Synthetic test record — VALID"
    ),
    (
        "C3456789", "RAHUL SINGH PATEL", "1995-08-30",
        "MUMBAI", "MUMBAI", "2020-01-15", "2030-01-14",
        "M", "IND", "P", "BV2024003", "2025-11-30",
        "VALID", "data/samples/portraits/C3456789.jpg", "Synthetic test record — VALID"
    ),
    (
        "D4567890", "SUNITA DEVI VERMA", "1978-03-12",
        "KOLKATA", "KOLKATA", "2018-09-05", "2028-09-04",
        "F", "IND", "P", None, None,
        "VALID", "data/samples/portraits/D4567890.jpg", "Synthetic test record — VALID (no visa)"
    ),
    (
        "E5678901", "VIKRAM NAIR KRISHNAN", "2000-12-01",
        "KOCHI", "KOCHI", "2022-04-18", "2032-04-17",
        "M", "IND", "P", "BV2024005", "2027-03-31",
        "VALID", "data/samples/portraits/E5678901.jpg", "Synthetic test record — VALID"
    ),
    # ── EXPIRED records ──────────────────────────────────────────────────────
    (
        "F6789012", "ANANYA ROY CHATTERJEE", "1980-07-04",
        "KOLKATA", "KOLKATA", "2013-02-28", "2023-02-27",
        "F", "IND", "P", None, None,
        "EXPIRED", "data/samples/portraits/F6789012.jpg", "Synthetic test record — EXPIRED passport"
    ),
    (
        "G7890123", "DEEPAK GUPTA MISHRA", "1970-09-19",
        "LUCKNOW", "LUCKNOW", "2012-11-11", "2022-11-10",
        "M", "IND", "P", None, None,
        "EXPIRED", "data/samples/portraits/G7890123.jpg", "Synthetic test record — EXPIRED passport"
    ),
    # ── FLAGGED_FOR_REVIEW records ────────────────────────────────────────────
    (
        "H8901234", "KAVITA SHARMA JOSHI", "1992-01-25",
        "JAIPUR", "JAIPUR", "2020-07-07", "2030-07-06",
        "F", "IND", "P", "BV2024008", "2026-01-31",
        "FLAGGED_FOR_REVIEW", "data/samples/portraits/H8901234.jpg", "Synthetic test record — inconsistent travel history"
    ),
    (
        "J0123456", "MOHAN DAS MALHOTRA", "1965-06-30",
        "PUNE", "PUNE", "2019-12-01", "2029-11-30",
        "M", "IND", "P", None, None,
        "FLAGGED_FOR_REVIEW", "data/samples/portraits/J0123456.jpg", "Synthetic test record — flagged for manual review"
    ),
    # ── INVALID record ───────────────────────────────────────────────────────
    (
        "K1234560", "TEST INVALID RECORD", "1999-03-03",
        "HYDERABAD", "HYDERABAD", "2021-03-03", "2031-03-02",
        "M", "IND", "P", None, None,
        "INVALID", None, "Synthetic test record — marked invalid for demo"
    ),
]


def seed_database() -> None:
    """
    Seed the database with synthetic prototype records.
    Idempotent — INSERT OR IGNORE prevents duplicate entries.
    """
    initialize_database()

    inserted = 0
    skipped = 0

    with db_session() as conn:
        for record in SYNTHETIC_RECORDS:
            cursor = conn.execute(INSERT_PASSPORT_RECORD, record)
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

    logger.info(
        "Database seeding complete — inserted: %d, skipped (already exist): %d",
        inserted,
        skipped,
    )
    print(f"\n✅ Database seeded successfully.")
    print(f"   Records inserted : {inserted}")
    print(f"   Records skipped  : {skipped} (already existed)")
    print(f"   Total records    : {len(SYNTHETIC_RECORDS)}")
