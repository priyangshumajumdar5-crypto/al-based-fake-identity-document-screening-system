"""
Individual validation rule functions for Module 2.

Each rule is independent and returns a CheckResult dict:
    {
        "status": "PASS" | "FAIL" | "NOT_AVAILABLE",
        "reason": "...",
    }

Rules never raise exceptions — they return FAIL with an explanation.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging_config import get_logger
from app.utils.date_utils import parse_date, is_expired, dates_are_consistent, normalise_date

logger = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
NOT_AVAILABLE = "NOT_AVAILABLE"


def _check(status: str, reason: str = "") -> Dict[str, str]:
    return {"status": status, "reason": reason}


def _get_value(fields: Dict[str, Any], key: str) -> str:
    """Extract the string value from a field dict or plain string."""
    f = fields.get(key)
    if f is None:
        return ""
    if isinstance(f, dict):
        return str(f.get("value", "")).strip()
    return str(f).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1: Required fields
# ─────────────────────────────────────────────────────────────────────────────

def validate_required_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that all required fields are present and non-empty.

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult with list of missing fields in reason.
    """
    missing: List[str] = []
    for key in settings.REQUIRED_FIELDS:
        val = _get_value(fields, key)
        if not val:
            missing.append(key)

    if missing:
        return _check(FAIL, f"Missing required fields: {', '.join(missing)}")
    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2: Date format validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_date_formats(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that date fields contain parseable, non-impossible dates.

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult.
    """
    date_fields = ["date_of_birth", "date_of_issue", "date_of_expiry"]
    invalid: List[str] = []

    for key in date_fields:
        raw = _get_value(fields, key)
        if not raw:
            continue  # Missing date handled by required_fields rule

        parsed, _ = parse_date(raw)
        if parsed is None:
            invalid.append(f"{key}='{raw}' (unparseable)")
        elif parsed > date(2100, 12, 31) or parsed < date(1900, 1, 1):
            invalid.append(f"{key}='{raw}' (out of plausible range)")

    if invalid:
        return _check(FAIL, f"Invalid date(s): {'; '.join(invalid)}")
    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3: Passport number format
# ─────────────────────────────────────────────────────────────────────────────

def validate_passport_number(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate passport number format against the configured pattern.

    The pattern is configurable via settings.PASSPORT_NUMBER_PATTERN
    so it can be adjusted for different document types without code changes.

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult.
    """
    pn = _get_value(fields, "passport_number")

    if not pn:
        return _check(NOT_AVAILABLE, "Passport number not extracted.")

    pattern = re.compile(settings.PASSPORT_NUMBER_PATTERN)
    if not pattern.match(pn):
        return _check(
            FAIL,
            f"Passport number '{pn}' does not match expected pattern "
            f"(pattern: {settings.PASSPORT_NUMBER_PATTERN}).",
        )

    length = len(pn)
    if not (settings.PASSPORT_NUMBER_MIN_LEN <= length <= settings.PASSPORT_NUMBER_MAX_LEN):
        return _check(
            FAIL,
            f"Passport number length {length} is outside expected range "
            f"[{settings.PASSPORT_NUMBER_MIN_LEN}–{settings.PASSPORT_NUMBER_MAX_LEN}].",
        )

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4: Passport expiry
# ─────────────────────────────────────────────────────────────────────────────

def validate_expiry(
    fields: Dict[str, Any],
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Check whether the passport has expired.

    Uses today's date as the reference unless override is provided.

    Args:
        fields:         Extracted document fields dict.
        reference_date: Optional override for "today".

    Returns:
        CheckResult with status PASS, FAIL (expired), or NOT_AVAILABLE.
    """
    expiry_raw = _get_value(fields, "date_of_expiry")

    if not expiry_raw:
        return _check(NOT_AVAILABLE, "Expiry date not available.")

    expired, message = is_expired(expiry_raw, reference_date)

    if message == "INVALID_DATE":
        return _check(FAIL, f"Expiry date '{expiry_raw}' could not be parsed.")

    if expired:
        return _check(FAIL, f"Passport is expired. {message}")

    return _check(PASS, message)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 5: Country code
# ─────────────────────────────────────────────────────────────────────────────

def validate_country_code(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    For this India→Bangladesh prototype, verify country code is IND.

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult.
    """
    cc = _get_value(fields, "country_code").upper().replace("INDIA", "IND")

    if not cc:
        return _check(NOT_AVAILABLE, "Country code not extracted.")

    if cc != "IND":
        return _check(
            FAIL,
            f"Country code '{cc}' does not match expected 'IND' for Indian passport.",
        )

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 6: MRZ check digits
# ─────────────────────────────────────────────────────────────────────────────

def validate_mrz_check_digits(mrz: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate ICAO 9303 MRZ check digits.

    Args:
        mrz: Parsed MRZ dict (from mrz_parser.parse_mrz).

    Returns:
        CheckResult.
    """
    if not mrz:
        return _check(NOT_AVAILABLE, "MRZ not available.")

    if mrz.get("parse_errors"):
        return _check(
            FAIL,
            f"MRZ parse errors: {'; '.join(mrz['parse_errors'])}",
        )

    if not mrz.get("line1") and not mrz.get("line2"):
        return _check(NOT_AVAILABLE, "MRZ lines are empty.")

    all_valid = mrz.get("all_check_digits_valid", False)
    if not all_valid:
        # Find which check digits failed
        failed = []
        check_digits = mrz.get("check_digits", {})
        for field_name, cd in check_digits.items():
            if isinstance(cd, dict) and not cd.get("valid", True):
                failed.append(
                    f"{field_name} (found '{cd.get('value')}', "
                    f"expected '{cd.get('expected')}')"
                )
        return _check(FAIL, f"MRZ check digit failures: {', '.join(failed)}")

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers (used by validate_passport flow)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Strip whitespace and uppercase for case-insensitive comparison."""
    if not s:
        return ""
    return " ".join(s.strip().upper().split())


def normalize_date(s: str) -> str:
    """
    Normalize a date string to DD/MM/YYYY canonical form.

    Handles:
        - DD/MM/YYYY (passthrough)
        - YYYY-MM-DD (ISO → DD/MM/YYYY)
        - DD-MM-YYYY or DD.MM.YYYY
    Returns the original string if parsing fails.
    """
    if not s or not s.strip():
        return ""
    s = s.strip()

    # Already DD/MM/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # ISO: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"

    # DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r"^(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})$", s)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    return s


def compare_field(ocr_val: str, db_val: str, field_type: str = "exact") -> str:
    """
    Compare an OCR-extracted value against a database value.

    Args:
        ocr_val:    Value from OCR extraction.
        db_val:     Value from database record.
        field_type: One of "name", "date", "exact".

    Returns:
        "MATCHED" | "MISMATCH" | "NOT_VERIFIED"
    """
    if not ocr_val or not ocr_val.strip():
        return "NOT_VERIFIED"
    if not db_val or not db_val.strip():
        return "NOT_VERIFIED"

    if field_type == "name":
        try:
            from rapidfuzz import fuzz
            score = fuzz.token_sort_ratio(
                normalize_text(ocr_val),
                normalize_text(db_val),
            )
            return "MATCHED" if score >= 90 else "MISMATCH"
        except ImportError:
            # Fallback to simple comparison if rapidfuzz not installed
            return "MATCHED" if normalize_text(ocr_val) == normalize_text(db_val) else "MISMATCH"

    elif field_type == "date":
        return (
            "MATCHED"
            if normalize_date(ocr_val) == normalize_date(db_val)
            else "MISMATCH"
        )

    else:  # exact
        return (
            "MATCHED"
            if normalize_text(ocr_val) == normalize_text(db_val)
            else "MISMATCH"
        )

