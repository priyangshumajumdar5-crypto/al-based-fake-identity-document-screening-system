"""
Module 4 — Passport Record Lookup.

Queries SQLite database for passport records using passport_number.
Checks both `passports` (CSV-seeded) and `passport_records` (synthetic DDL) tables.

Distinguishes 4 statuses:
  - FOUND
  - NOT_FOUND
  - MISSING_PASSPORT_NUMBER
  - DATABASE_ERROR

Important: Failure to find a passport record does NOT prevent face verification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.core.logging_config import get_logger
from app.database.database import db_session
from app.database.models import SELECT_BY_PASSPORT_NUMBER, SELECT_FROM_PASSPORTS

logger = get_logger(__name__)

# Status constants
DB_STATUS_FOUND = "FOUND"
DB_STATUS_NOT_FOUND = "NOT_FOUND"
DB_STATUS_MISSING_PN = "MISSING_PASSPORT_NUMBER"
DB_STATUS_ERROR = "DATABASE_ERROR"


def lookup_passport_record(
    passport_number: Optional[str],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Look up passport number in the SQLite database.

    Args:
        passport_number: Extracted or provided passport number.

    Returns:
        (status, record_dict, table_source, error_message)
        - status: FOUND, NOT_FOUND, MISSING_PASSPORT_NUMBER, or DATABASE_ERROR
        - record_dict: Dict of database fields if found, else None
        - table_source: Name of table where record was found ('passports' or 'passport_records'), else None
        - error_message: Exception message if DATABASE_ERROR, else None
    """
    if not passport_number or not str(passport_number).strip():
        logger.info("Passport lookup skipped: passport_number is missing or empty.")
        return DB_STATUS_MISSING_PN, None, None, None

    pn = str(passport_number).strip().upper()
    logger.info("Passport lookup starting for passport_number='%s'", pn)

    try:
        with db_session() as conn:
            # 1. Try 'passports' table (CSV-seeded records)
            try:
                row = conn.execute(SELECT_FROM_PASSPORTS, (pn,)).fetchone()
                if row:
                    record = dict(row)
                    logger.info("Passport record found in 'passports' table for '%s'", pn)
                    return DB_STATUS_FOUND, record, "passports", None
            except Exception as e_passports:
                logger.debug("Query on 'passports' table failed or table missing: %s", e_passports)

            # 2. Try 'passport_records' table (synthetic DDL records)
            try:
                row = conn.execute(SELECT_BY_PASSPORT_NUMBER, (pn,)).fetchone()
                if row:
                    record = dict(row)
                    logger.info("Passport record found in 'passport_records' table for '%s'", pn)
                    return DB_STATUS_FOUND, record, "passport_records", None
            except Exception as e_records:
                logger.debug("Query on 'passport_records' table failed: %s", e_records)

            logger.info("No passport record found in database for '%s'", pn)
            return DB_STATUS_NOT_FOUND, None, None, None

    except Exception as exc:
        logger.error("Database error during passport lookup for '%s': %s", pn, exc)
        return DB_STATUS_ERROR, None, None, str(exc)
