"""
Module 4 — Issue Date Resolution and Elapsed Time Calculation.

Implements source priority policy:
  1. Database record issue date (if matching DB record found and date is valid)
  2. OCR issue date (if reliable date extracted)
  3. Unavailable (if neither source provides a valid date)

Validates dates against future/invalid dates and calculates elapsed time
relative to a configurable reference date.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

from app.core.logging_config import get_logger
from app.utils.date_utils import normalise_date, parse_date

logger = get_logger(__name__)

# Source constants
SOURCE_DB = "DATABASE"
SOURCE_OCR = "OCR"
SOURCE_UNAVAILABLE = "UNAVAILABLE"


def resolve_issue_date(
    db_record: Optional[Dict[str, Any]] = None,
    ocr_issue_date: Optional[str] = None,
    reference_date: Optional[date] = None,
) -> Tuple[Optional[date], str, str, Optional[float], Optional[int], Optional[str]]:
    """
    Resolve the passport issue date following documented priority policy.

    Priority:
        1. DB issue date (if valid)
        2. OCR issue date (if valid)
        3. Unavailable

    Args:
        db_record: Database record dictionary (if found).
        ocr_issue_date: Issue date extracted from OCR.
        reference_date: Date to calculate elapsed time against (defaults to date.today()).

    Returns:
        (resolved_date, date_str, source, elapsed_years, elapsed_days, warning_msg)
        - resolved_date: datetime.date object or None
        - date_str: Normalized YYYY-MM-DD string or 'UNAVAILABLE'
        - source: 'DATABASE', 'OCR', or 'UNAVAILABLE'
        - elapsed_years: Float elapsed years from issue date to reference date, or None
        - elapsed_days: Int elapsed days, or None
        - warning_msg: Warning message regarding conflict/future/invalid date, or None
    """
    ref_date = reference_date or date.today()

    db_date_raw: Optional[str] = None
    if db_record and isinstance(db_record, dict):
        db_date_raw = str(db_record.get("date_of_issue") or "").strip()

    ocr_date_raw = str(ocr_issue_date or "").strip()

    parsed_db_date, norm_db_date = parse_date(db_date_raw) if db_date_raw else (None, "")
    parsed_ocr_date, norm_ocr_date = parse_date(ocr_date_raw) if ocr_date_raw else (None, "")

    warning: Optional[str] = None

    # Check for discrepancy if both sources provided valid dates
    if parsed_db_date and parsed_ocr_date and norm_db_date != norm_ocr_date:
        warning = (
            f"Date discrepancy detected: DB issue date ('{norm_db_date}') differs "
            f"from OCR issue date ('{norm_ocr_date}'). Prioritising DB issue date per policy."
        )
        logger.warning(warning)

    resolved_date: Optional[date] = None
    date_str: str = SOURCE_UNAVAILABLE
    source: str = SOURCE_UNAVAILABLE

    # Resolve according to priority policy
    if parsed_db_date:
        resolved_date = parsed_db_date
        date_str = norm_db_date
        source = SOURCE_DB
    elif parsed_ocr_date:
        resolved_date = parsed_ocr_date
        date_str = norm_ocr_date
        source = SOURCE_OCR
    else:
        source = SOURCE_UNAVAILABLE
        date_str = "UNAVAILABLE"

    if resolved_date is None:
        return None, "UNAVAILABLE", SOURCE_UNAVAILABLE, None, None, warning

    # Check if resolved issue date is in the future relative to reference date
    if resolved_date > ref_date:
        future_warn = (
            f"Invalid issue date: Resolved issue date '{date_str}' is in the future "
            f"relative to reference date '{ref_date.isoformat()}'. Elapsed time cannot be computed."
        )
        logger.warning(future_warn)
        full_warn = f"{warning} | {future_warn}" if warning else future_warn
        return resolved_date, date_str, source, None, None, full_warn

    # Compute elapsed time
    delta = ref_date - resolved_date
    elapsed_days = delta.days
    elapsed_years = round(elapsed_days / 365.25, 2)

    return resolved_date, date_str, source, elapsed_years, elapsed_days, warning
