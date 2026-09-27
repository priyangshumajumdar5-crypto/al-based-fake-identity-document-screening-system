"""
Pydantic schemas for Module 2 — Validation responses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    """Result for a single validation check."""

    status: str = Field(description="PASS | FAIL | NOT_AVAILABLE")
    reason: str = Field(default="", description="Human-readable explanation.")


class DatabaseFieldComparison(BaseModel):
    """Comparison of a single field between OCR output and database record."""

    ocr_value: str = Field(default="", description="Value extracted by OCR.")
    db_value: str = Field(default="", description="Value from database record.")
    match: str = Field(default="NOT_AVAILABLE", description="MATCH | MISMATCH | NOT_AVAILABLE")


class DatabaseComparisonResult(BaseModel):
    """Full OCR-vs-database comparison for a passport record."""

    record_found: bool = Field(default=False, description="Whether a DB record was found.")
    db_status: str = Field(default="", description="Status field from database (VALID/EXPIRED/etc).")
    field_comparisons: Dict[str, DatabaseFieldComparison] = Field(
        default_factory=dict,
        description="Per-field comparison results.",
    )
    overall_match: str = Field(
        default="NOT_AVAILABLE",
        description="MATCH (all compared fields match) | MISMATCH | NOT_FOUND | NOT_AVAILABLE",
    )


class ValidationChecks(BaseModel):
    """All individual validation check results."""

    required_fields: CheckResult = Field(default_factory=CheckResult)
    date_format: CheckResult = Field(default_factory=CheckResult)
    passport_number_format: CheckResult = Field(default_factory=CheckResult)
    passport_expiry: CheckResult = Field(default_factory=CheckResult)
    country_code: CheckResult = Field(default_factory=CheckResult)
    cross_field_consistency: CheckResult = Field(default_factory=CheckResult)
    mrz_check_digits: CheckResult = Field(default_factory=CheckResult)
    database_match: CheckResult = Field(default_factory=CheckResult)


class ValidationResponse(BaseModel):
    """Complete Module 2 validation result."""

    validation_status: str = Field(
        description="Overall status: PASS | FAIL | REVIEW | ERROR"
    )
    checks: ValidationChecks = Field(default_factory=ValidationChecks)
    database_record_found: bool = Field(default=False)
    database_comparison: Optional[DatabaseComparisonResult] = Field(default=None)
    discrepancies: List[str] = Field(
        default_factory=list,
        description="List of specific discrepancy descriptions.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings that warrant human review.",
    )
    recommendation: str = Field(
        default="",
        description="Human-readable recommendation for the officer.",
    )
    error: Optional[str] = Field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 v2 — Per-field OCR-vs-DB comparison models
# Used by validate_passport() (new flow for real CSV-seeded database)
# ─────────────────────────────────────────────────────────────────────────────

class FieldComparison(BaseModel):
    """Comparison result for a single passport field."""

    field_name: str = Field(description="Name of the compared field.")
    ocr_value: str = Field(default="", description="Value extracted by OCR.")
    db_value: str = Field(default="", description="Value from database record.")
    status: str = Field(
        default="NOT_VERIFIED",
        description="MATCHED | MISMATCH | NOT_VERIFIED",
    )


class ValidationSummary(BaseModel):
    """Aggregate counts of field comparison outcomes."""

    matched_count: int = Field(default=0)
    mismatch_count: int = Field(default=0)
    not_verified_count: int = Field(default=0)


class ValidationResult(BaseModel):
    """
    Full OCR-vs-database validation result for validate_passport() flow.
    Designed for the Streamlit officer dashboard.
    """

    passport_number: str = Field(default="", description="Passport number used for lookup.")
    db_match: bool = Field(default=False, description="Whether any DB record was found.")
    fuzzy_matched: bool = Field(
        default=False,
        description="True if match was fuzzy (not exact passport number).",
    )
    field_comparisons: List[FieldComparison] = Field(
        default_factory=list,
        description="Per-field comparison results.",
    )
    summary: ValidationSummary = Field(default_factory=ValidationSummary)
    overall_status: str = Field(
        default="NOT_FOUND",
        description="VERIFIED | NEEDS_REVIEW | NOT_FOUND",
    )
    passport_photo_path: Optional[str] = Field(
        default=None,
        description="Path to the DB reference photo for side-by-side display.",
    )
    message: str = Field(
        default="",
        description="Human-readable status message for the officer.",
    )

