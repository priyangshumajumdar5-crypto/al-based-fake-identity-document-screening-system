"""
Date parsing, validation, and normalisation utilities.
All dates are normalised to ISO 8601 (YYYY-MM-DD) internally.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional, Tuple

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Date format patterns tried in order during parsing
DATE_FORMATS = [
    "%Y-%m-%d",   # 1990-05-15 (ISO, preferred)
    "%d/%m/%Y",   # 15/05/1990 (common Indian format)
    "%d-%m-%Y",   # 15-05-1990
    "%d %b %Y",   # 15 MAY 1990
    "%d %B %Y",   # 15 May 1990
    "%d/%m/%y",   # 15/05/90
    "%d-%m-%y",   # 15-05-90
    "%B %d, %Y",  # May 15, 1990
    "%b %d, %Y",  # May 15, 1990
]


def parse_date(raw: str) -> Tuple[Optional[date], str]:
    """
    Attempt to parse a date string using multiple known formats.

    Args:
        raw: Raw date string from OCR or user input.

    Returns:
        (parsed_date, normalised_string) — or (None, "") on failure.
    """
    if not raw or not raw.strip():
        return None, ""

    cleaned = raw.strip().upper()
    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Try MRZ format YYMMDD first if it looks like 6 digits
    mrz_match = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", cleaned)
    if mrz_match:
        return parse_mrz_date(cleaned)

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            # Validate leap-year (e.g. 29 Feb on non-leap year is caught by strptime)
            return dt.date(), dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    logger.debug("Unable to parse date: %r", raw)
    return None, ""


def parse_mrz_date(yymmdd: str) -> Tuple[Optional[date], str]:
    """
    Parse an MRZ-format date (YYMMDD) to a full date.

    Years 00–29 are treated as 2000–2029.
    Years 30–99 are treated as 1930–1999.

    Args:
        yymmdd: Six-character MRZ date string.

    Returns:
        (parsed_date, YYYY-MM-DD string) or (None, "") on failure.
    """
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return None, ""

    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = (2000 + yy) if yy <= 29 else (1900 + yy)

    try:
        d = date(year, mm, dd)
        return d, d.strftime("%Y-%m-%d")
    except ValueError:
        return None, ""


def date_to_mrz(d: date) -> str:
    """
    Convert a date object to MRZ YYMMDD format.

    Args:
        d: A date object.

    Returns:
        Six-character MRZ date string.
    """
    return d.strftime("%y%m%d")


def is_expired(expiry_date_str: str, reference_date: Optional[date] = None) -> Tuple[bool, str]:
    """
    Determine whether a passport has expired.

    Args:
        expiry_date_str: Expiry date in any recognised format.
        reference_date: Date to compare against (defaults to today).

    Returns:
        (is_expired, message) tuple.
    """
    ref = reference_date or date.today()
    parsed, _ = parse_date(expiry_date_str)

    if parsed is None:
        return False, "INVALID_DATE"

    if parsed < ref:
        return True, f"EXPIRED (expired {parsed.isoformat()})"

    return False, f"VALID (expires {parsed.isoformat()})"


def dates_are_consistent(earlier_str: str, later_str: str) -> Tuple[bool, str]:
    """
    Assert that earlier_str represents a date strictly before later_str.

    Args:
        earlier_str: The date expected to be earlier.
        later_str:   The date expected to be later.

    Returns:
        (consistent, message) tuple.
    """
    d1, _ = parse_date(earlier_str)
    d2, _ = parse_date(later_str)

    if d1 is None or d2 is None:
        return False, "One or both dates could not be parsed."

    if d1 >= d2:
        return False, f"{d1.isoformat()} is not before {d2.isoformat()}"

    return True, "Consistent"


def normalise_date(raw: str) -> str:
    """
    Normalise a date string to YYYY-MM-DD.

    Args:
        raw: Date string in any recognised format.

    Returns:
        YYYY-MM-DD string or empty string on failure.
    """
    _, normalised = parse_date(raw)
    return normalised
