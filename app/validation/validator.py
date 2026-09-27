"""
Module 2 — Document Validation orchestrator.

Combines all individual rules, runs the database lookup,
and produces a structured ValidationResponse.

Architecture:
    validate_document(ocr_result)
        → validate_required_fields()
        → validate_date_formats()
        → validate_passport_number()
        → validate_expiry()
        → validate_country_code()
        → validate_cross_field_consistency()
        → validate_mrz_check_digits()
        → validate_database_record()   ← queries SQLite
        → ValidationResponse

This module NEVER mixes OCR logic with validation logic.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_config import get_logger
from app.database.database import db_session
from app.database.models import SELECT_BY_PASSPORT_NUMBER
from app.utils.date_utils import normalise_date
from app.validation.consistency import validate_cross_field_consistency
from app.validation.rules import (
    validate_country_code,
    validate_date_formats,
    validate_expiry,
    validate_mrz_check_digits,
    validate_passport_number,
    validate_required_fields,
)
from app.validation.schemas import (
    CheckResult,
    DatabaseComparisonResult,
    DatabaseFieldComparison,
    ValidationChecks,
    ValidationResponse,
)

logger = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
NOT_AVAILABLE = "NOT_AVAILABLE"
REVIEW = "REVIEW"


def _check_result(d: Dict[str, str]) -> CheckResult:
    return CheckResult(status=d.get("status", NOT_AVAILABLE), reason=d.get("reason", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Database comparison
# ─────────────────────────────────────────────────────────────────────────────

def _get_value(fields: Dict[str, Any], key: str) -> str:
    """Extract string value from a field dict or plain string."""
    f = fields.get(key)
    if f is None:
        return ""
    if isinstance(f, dict):
        return str(f.get("value", "")).strip()
    return str(f).strip()


def _compare_field(ocr_val: str, db_val: str) -> str:
    """Return MATCH / MISMATCH / NOT_AVAILABLE."""
    if not ocr_val or not db_val:
        return NOT_AVAILABLE
    return "MATCH" if ocr_val.strip().upper() == db_val.strip().upper() else "MISMATCH"


def validate_database_record(fields: Dict[str, Any]) -> Tuple[CheckResult, Optional[DatabaseComparisonResult]]:
    """
    Look up passport number in the SQLite database and compare fields.

    If the passport is not found in the prototype database, the result
    is NOT_FOUND with recommendation for manual review.
    The system does NOT automatically declare a passport fraudulent.

    Args:
        fields: Extracted document fields dict.

    Returns:
        (CheckResult, DatabaseComparisonResult | None)
    """
    passport_number = _get_value(fields, "passport_number")

    if not passport_number:
        return (
            CheckResult(status=NOT_AVAILABLE, reason="Passport number not available for DB lookup."),
            None,
        )

    logger.info("Database lookup: passport_number=%s", passport_number)

    try:
        with db_session() as conn:
            row = conn.execute(SELECT_BY_PASSPORT_NUMBER, (passport_number,)).fetchone()
    except Exception as exc:
        logger.error("Database error during lookup: %s", exc)
        return (
            CheckResult(status=FAIL, reason=f"Database error: {exc}"),
            None,
        )

    if row is None:
        logger.info("Passport not found in database: %s", passport_number)
        return (
            CheckResult(
                status=REVIEW,
                reason=(
                    f"Passport number '{passport_number}' not found in prototype database. "
                    "Record may be valid but not in the controlled test dataset. "
                    "Manual review recommended."
                ),
            ),
            DatabaseComparisonResult(
                record_found=False,
                overall_match="NOT_FOUND",
            ),
        )

    db_record = dict(row)
    db_status = db_record.get("status", "")
    logger.info("Database record found — status: %s", db_status)

    # Compare OCR fields against database fields
    fields_to_compare = {
        "passport_number": "passport_number",
        "name":            "name",
        "date_of_birth":   "date_of_birth",
        "date_of_expiry":  "date_of_expiry",
        "date_of_issue":   "date_of_issue",
        "sex":             "sex",
        "country_code":    "country_code",
    }

    comparisons: Dict[str, DatabaseFieldComparison] = {}
    mismatches: List[str] = []

    for ocr_key, db_key in fields_to_compare.items():
        ocr_val = _get_value(fields, ocr_key)
        db_val = str(db_record.get(db_key, ""))

        # Normalise dates before comparison
        if "date" in ocr_key:
            ocr_val = normalise_date(ocr_val) or ocr_val
            db_val = normalise_date(db_val) or db_val

        match_status = _compare_field(ocr_val, db_val)
        comparisons[ocr_key] = DatabaseFieldComparison(
            ocr_value=ocr_val,
            db_value=db_val,
            match=match_status,
        )
        if match_status == "MISMATCH":
            mismatches.append(ocr_key)

    # Determine overall match
    compared = [c for c in comparisons.values() if c.match != NOT_AVAILABLE]
    overall: str
    if not compared:
        overall = NOT_AVAILABLE
    elif mismatches:
        overall = "MISMATCH"
    else:
        overall = "MATCH"

    db_result = DatabaseComparisonResult(
        record_found=True,
        db_status=db_status,
        field_comparisons=comparisons,
        overall_match=overall,
    )

    # Determine check status
    if db_status in ("EXPIRED", "INVALID", "FLAGGED_FOR_REVIEW"):
        check_status = FAIL
        reason = f"Database record status: {db_status}."
        if mismatches:
            reason += f" Field mismatches: {', '.join(mismatches)}."
    elif mismatches:
        check_status = FAIL
        reason = f"OCR data mismatches database on: {', '.join(mismatches)}."
    else:
        check_status = PASS
        reason = f"Database record found and fields match. Status: {db_status}."

    return CheckResult(status=check_status, reason=reason), db_result


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def validate_document(
    fields: Dict[str, Any],
    mrz: Optional[Dict[str, Any]] = None,
) -> ValidationResponse:
    """
    Run all validation checks and return a structured ValidationResponse.

    This is the primary entry point for Module 2.

    Args:
        fields: Extracted visible-page fields from OCR.
        mrz:    Parsed MRZ dict from mrz_parser (optional).

    Returns:
        ValidationResponse with all check results.
    """
    logger.info("Starting document validation.")

    discrepancies: List[str] = []
    warnings: List[str] = []

    # ── Run individual rule checks ─────────────────────────────────────────────
    r_required   = validate_required_fields(fields)
    r_dates      = validate_date_formats(fields)
    r_pn         = validate_passport_number(fields)
    r_expiry     = validate_expiry(fields)
    r_country    = validate_country_code(fields)
    r_mrz_cd     = validate_mrz_check_digits(mrz)

    # MRZ consistency (from mrz field if available)
    mrz_consistency = mrz.get("consistency") if mrz else None
    r_consistency = validate_cross_field_consistency(fields, mrz_consistency)

    # Database lookup
    r_db_check, db_comparison = validate_database_record(fields)
    # Convert CheckResult to dict for uniform access with other check results
    r_db_check_dict = {"status": r_db_check.status, "reason": r_db_check.reason}

    # ── Collect discrepancies ─────────────────────────────────────────────────
    all_checks = {
        "required_fields":        r_required,
        "date_format":            r_dates,
        "passport_number_format": r_pn,
        "passport_expiry":        r_expiry,
        "country_code":           r_country,
        "cross_field_consistency":r_consistency,
        "mrz_check_digits":       r_mrz_cd,
        "database_match":         r_db_check_dict,
    }

    for check_name, result in all_checks.items():
        if result["status"] == FAIL and result.get("reason"):
            discrepancies.append(f"{check_name}: {result['reason']}")
        elif result["status"] == REVIEW and result.get("reason"):
            warnings.append(f"{check_name}: {result['reason']}")
        elif result["status"] == NOT_AVAILABLE:
            warnings.append(f"{check_name}: Not available — {result.get('reason', '')}")

    # ── Determine overall status ──────────────────────────────────────────────
    check_statuses = [r["status"] for r in all_checks.values()]

    if FAIL in check_statuses:
        overall_status = FAIL
        recommendation = (
            "Document validation FAILED. Manual review by authorised officer required. "
            "Do not allow entry based on automated screening alone."
        )
    elif REVIEW in check_statuses:
        overall_status = REVIEW
        recommendation = (
            "Document requires manual review. Automated screening could not fully verify "
            "all aspects of this document."
        )
    elif all(s in (PASS, NOT_AVAILABLE) for s in check_statuses):
        if any(s == NOT_AVAILABLE for s in check_statuses):
            overall_status = REVIEW
            recommendation = (
                "Validation partially successful. Some checks returned NOT_AVAILABLE. "
                "Officer should verify missing information manually."
            )
        else:
            overall_status = PASS
            recommendation = "All automated checks passed. Final decision rests with the authorised officer."
    else:
        overall_status = REVIEW
        recommendation = "Screening inconclusive. Manual review by authorised officer required."

    logger.info("Validation complete — status: %s", overall_status)

    checks = ValidationChecks(
        required_fields          =_check_result(r_required),
        date_format              =_check_result(r_dates),
        passport_number_format   =_check_result(r_pn),
        passport_expiry          =_check_result(r_expiry),
        country_code             =_check_result(r_country),
        cross_field_consistency  =_check_result(r_consistency),
        mrz_check_digits         =_check_result(r_mrz_cd),
        database_match           =_check_result(r_db_check_dict),
    )

    return ValidationResponse(
        validation_status=overall_status,
        checks=checks,
        database_record_found=db_comparison.record_found if db_comparison else False,
        database_comparison=db_comparison,
        discrepancies=discrepancies,
        warnings=warnings,
        recommendation=recommendation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# New validate_passport() — OCR→DB per-field comparison
# ─────────────────────────────────────────────────────────────────────────────

def validate_passport(ocr_result: Dict[str, Any]) -> "ValidationResult":
    """
    Validate a passport by comparing OCR-extracted data against the real
    CSV-seeded SQLite database (table `passports`).

    Steps:
        1. Extract passport_number from ocr_result
        2. Look up in DB (exact match, then fuzzy fallback)
        3. Compare fields: surname, given_name, date_of_birth, date_of_expiry,
           place_of_birth, place_of_issue, date_of_issue, gender
        4. Return ValidationResult with per-field comparison + overall status

    Args:
        ocr_result: Dict of OCR fields (each field is either a string or
                    a dict with "value" key).

    Returns:
        ValidationResult with field_comparisons, summary, and overall_status.
    """
    from app.validation.consistency import lookup_passport
    from app.validation.rules import compare_field, normalize_date as norm_date_fn
    from app.validation.schemas import (
        FieldComparison,
        ValidationResult,
        ValidationSummary,
    )

    def _val(key: str) -> str:
        """Extract string value from OCR result field."""
        f = ocr_result.get(key)
        if f is None:
            return ""
        if isinstance(f, dict):
            return str(f.get("value", "")).strip()
        return str(f).strip()

    # 1. Extract passport number
    passport_number = _val("passport_number")
    if not passport_number:
        logger.info("validate_passport: no passport_number in OCR result")
        return ValidationResult(
            passport_number="",
            db_match=False,
            overall_status="NOT_FOUND",
            message="Passport number could not be extracted from the document. Full manual verification required.",
        )

    # 2. Database lookup
    db_record, fuzzy_matched = lookup_passport(passport_number)

    if db_record is None:
        logger.info("validate_passport: no DB match for '%s'", passport_number)
        return ValidationResult(
            passport_number=passport_number,
            db_match=False,
            overall_status="NOT_FOUND",
            message=f"No matching record found for passport number '{passport_number}'. Full manual verification required.",
        )

    logger.info(
        "validate_passport: DB match found (fuzzy=%s) for '%s'",
        fuzzy_matched, passport_number,
    )

    # 3. Compare fields
    # Map: (field_name_for_display, ocr_key, db_key, field_type)
    FIELD_MAP = [
        ("Surname",         "surname",        "surname",        "name"),
        ("Given Name",      "given_names",    "given_name",     "name"),
        ("Date of Birth",   "date_of_birth",  "date_of_birth",  "date"),
        ("Date of Expiry",  "date_of_expiry", "date_of_expiry", "date"),
        ("Place of Birth",  "place_of_birth", "place_of_birth", "name"),
        ("Place of Issue",  "place_of_issue", "place_of_issue", "name"),
        ("Date of Issue",   "date_of_issue",  "date_of_issue",  "date"),
        ("Gender",          "sex",            "gender",         "exact"),
    ]

    # If OCR returns combined "name" but not separate surname/given_names,
    # try to use the combined name for both surname and given_name comparison
    ocr_name = _val("name")
    ocr_surname = _val("surname")
    ocr_given = _val("given_names")

    # If we have a combined name but no separate fields, use it for comparison
    if ocr_name and not ocr_surname and not ocr_given:
        # Compare combined OCR name against combined DB name
        db_surname = str(db_record.get("surname", ""))
        db_given = str(db_record.get("given_name", ""))
        db_combined = f"{db_surname} {db_given}".strip()

        FIELD_MAP = [
            ("Full Name",       "_combined_name", "_combined_db_name", "name"),
            ("Date of Birth",   "date_of_birth",  "date_of_birth",    "date"),
            ("Date of Expiry",  "date_of_expiry", "date_of_expiry",   "date"),
            ("Place of Birth",  "place_of_birth", "place_of_birth",   "name"),
            ("Place of Issue",  "place_of_issue", "place_of_issue",   "name"),
            ("Date of Issue",   "date_of_issue",  "date_of_issue",    "date"),
            ("Gender",          "sex",            "gender",           "exact"),
        ]
        # Inject synthetic keys for comparison
        ocr_result = dict(ocr_result)  # shallow copy
        ocr_result["_combined_name"] = ocr_name
        db_record["_combined_db_name"] = db_combined

    comparisons = []
    matched = 0
    mismatched = 0
    not_verified = 0

    for display_name, ocr_key, db_key, field_type in FIELD_MAP:
        ocr_val = _val(ocr_key)
        db_val = str(db_record.get(db_key, ""))

        status = compare_field(ocr_val, db_val, field_type)

        comparisons.append(FieldComparison(
            field_name=display_name,
            ocr_value=ocr_val,
            db_value=db_val,
            status=status,
        ))

        if status == "MATCHED":
            matched += 1
        elif status == "MISMATCH":
            mismatched += 1
        else:
            not_verified += 1

    # 4. Overall status
    if mismatched == 0 and not_verified == 0:
        overall_status = "VERIFIED"
        message = "All fields match the database record. Document verified."
    else:
        overall_status = "NEEDS_REVIEW"
        parts = []
        if mismatched > 0:
            parts.append(f"{mismatched} field(s) do not match")
        if not_verified > 0:
            parts.append(f"{not_verified} field(s) could not be verified")
        message = f"Review required: {'; '.join(parts)}."

    if fuzzy_matched:
        message += " (Note: passport number was fuzzy-matched, not exact.)"

    # 5. Get reference photo path
    photo_path = db_record.get("passport_photo_path", None)

    return ValidationResult(
        passport_number=passport_number,
        db_match=True,
        fuzzy_matched=fuzzy_matched,
        field_comparisons=comparisons,
        summary=ValidationSummary(
            matched_count=matched,
            mismatch_count=mismatched,
            not_verified_count=not_verified,
        ),
        overall_status=overall_status,
        passport_photo_path=photo_path,
        message=message,
    )

