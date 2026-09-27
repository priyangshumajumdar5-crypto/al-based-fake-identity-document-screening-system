"""
Cross-field consistency checks for Module 2.

These checks verify the logical relationships between different
document fields rather than validating individual field formats.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging_config import get_logger
from app.utils.date_utils import dates_are_consistent, normalise_date, parse_date

logger = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
NOT_AVAILABLE = "NOT_AVAILABLE"


def _check(status: str, reason: str = "") -> Dict[str, str]:
    return {"status": status, "reason": reason}


def _get_value(fields: Dict[str, Any], key: str) -> str:
    f = fields.get(key)
    if f is None:
        return ""
    if isinstance(f, dict):
        return str(f.get("value", "")).strip()
    return str(f).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Consistency rule 1: Date ordering
# ─────────────────────────────────────────────────────────────────────────────

def check_date_ordering(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify logical date ordering:
        DOB < Date of Issue < Date of Expiry

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult.
    """
    dob  = _get_value(fields, "date_of_birth")
    doi  = _get_value(fields, "date_of_issue")
    doex = _get_value(fields, "date_of_expiry")

    failures: List[str] = []

    if dob and doi:
        ok, msg = dates_are_consistent(dob, doi)
        if not ok:
            failures.append(f"DOB ({dob}) is not before date of issue ({doi}): {msg}")

    if doi and doex:
        ok, msg = dates_are_consistent(doi, doex)
        if not ok:
            failures.append(f"Date of issue ({doi}) is not before date of expiry ({doex}): {msg}")

    if dob and doex:
        ok, msg = dates_are_consistent(dob, doex)
        if not ok:
            failures.append(f"DOB ({dob}) is not before date of expiry ({doex}): {msg}")

    if not dob and not doi and not doex:
        return _check(NOT_AVAILABLE, "No date fields available for ordering check.")

    if failures:
        return _check(FAIL, "; ".join(failures))

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency rule 2: MRZ vs visible-page fields
# ─────────────────────────────────────────────────────────────────────────────

def check_mrz_consistency(
    mrz_consistency: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Evaluate the MRZ vs visible field consistency result.

    Args:
        mrz_consistency: Dict from mrz_parser.check_mrz_consistency().
                         Values are MATCH / MISMATCH / NOT_AVAILABLE.

    Returns:
        CheckResult with mismatched fields listed in reason.
    """
    if not mrz_consistency:
        return _check(NOT_AVAILABLE, "MRZ consistency data not available.")

    mismatches = [
        f"{field}: MISMATCH"
        for field, status in mrz_consistency.items()
        if status == "MISMATCH"
    ]

    if mismatches:
        return _check(
            FAIL,
            f"MRZ fields do not match visible page — {'; '.join(mismatches)}. "
            f"Manual review recommended.",
        )

    # If everything is NOT_AVAILABLE, flag it
    all_na = all(s == "NOT_AVAILABLE" for s in mrz_consistency.values())
    if all_na:
        return _check(NOT_AVAILABLE, "All MRZ-to-visible comparisons returned NOT_AVAILABLE.")

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency rule 3: Passport type
# ─────────────────────────────────────────────────────────────────────────────

def check_passport_type(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify passport type is 'P' (ordinary passport) for this prototype.

    Args:
        fields: Extracted document fields dict.

    Returns:
        CheckResult.
    """
    pt = _get_value(fields, "passport_type").upper().strip()

    if not pt:
        return _check(NOT_AVAILABLE, "Passport type not extracted.")

    # Accept P, PA (diplomatic annex), PO, etc. — only flag non-P series
    if not pt.startswith("P"):
        return _check(
            FAIL,
            f"Unexpected passport type '{pt}'. Expected 'P' series for this prototype.",
        )

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate cross-field consistency check
# ─────────────────────────────────────────────────────────────────────────────

def validate_cross_field_consistency(
    fields: Dict[str, Any],
    mrz_consistency: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run all cross-field consistency checks and aggregate the result.

    Args:
        fields:          Extracted visible document fields.
        mrz_consistency: MRZ vs visible comparison dict.

    Returns:
        CheckResult. Status is FAIL if any sub-check fails.
    """
    sub_checks = {
        "date_ordering":     check_date_ordering(fields),
        "mrz_consistency":   check_mrz_consistency(mrz_consistency),
        "passport_type":     check_passport_type(fields),
    }

    failures = [
        f"{name}: {res['reason']}"
        for name, res in sub_checks.items()
        if res["status"] == FAIL
    ]

    if failures:
        return _check(FAIL, " | ".join(failures))

    return _check(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# Database lookup with fuzzy fallback (used by validate_passport flow)
# ─────────────────────────────────────────────────────────────────────────────

def lookup_passport(passport_number: str) -> tuple:
    """
    Look up a passport number in the `passports` table.

    1. Exact match first.
    2. If not found, fuzzy match against all passport_numbers (threshold 85).

    Args:
        passport_number: The passport number to look up.

    Returns:
        (db_record_dict | None, fuzzy_matched: bool)
    """
    from app.database.database import db_session
    from app.database.models import SELECT_FROM_PASSPORTS, SELECT_ALL_PASSPORT_NUMBERS

    if not passport_number or not passport_number.strip():
        return None, False

    pn = passport_number.strip().upper()
    logger.info("lookup_passport: searching for '%s'", pn)

    try:
        with db_session() as conn:
            # 1. Exact match
            row = conn.execute(SELECT_FROM_PASSPORTS, (pn,)).fetchone()
            if row:
                logger.info("lookup_passport: exact match found for '%s'", pn)
                return dict(row), False

            # 2. Fuzzy fallback
            all_rows = conn.execute(SELECT_ALL_PASSPORT_NUMBERS).fetchall()
            all_pns = [r[0] for r in all_rows if r[0]]

            if not all_pns:
                logger.info("lookup_passport: passports table is empty")
                return None, False

            try:
                from rapidfuzz import process, fuzz
                best = process.extractOne(
                    pn,
                    all_pns,
                    scorer=fuzz.ratio,
                    score_cutoff=85,
                )
                if best:
                    matched_pn, score, _ = best
                    logger.info(
                        "lookup_passport: fuzzy match '%s' → '%s' (score=%.1f)",
                        pn, matched_pn, score,
                    )
                    row = conn.execute(SELECT_FROM_PASSPORTS, (matched_pn,)).fetchone()
                    if row:
                        return dict(row), True
            except ImportError:
                logger.warning("rapidfuzz not installed — fuzzy matching disabled")

            logger.info("lookup_passport: no match found for '%s'", pn)
            return None, False

    except Exception as exc:
        logger.error("lookup_passport: database error: %s", exc)
        return None, False
