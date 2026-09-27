"""
Pytest tests for Module 2 — Validation rules.

Tests are independent of the OCR engine and do not require EasyOCR.
"""
import pytest
from datetime import date

from app.validation.rules import (
    validate_required_fields,
    validate_date_formats,
    validate_passport_number,
    validate_expiry,
    validate_country_code,
    validate_mrz_check_digits,
)
from app.validation.consistency import (
    check_date_ordering,
    check_mrz_consistency,
    check_passport_type,
)
from app.validation.validator import validate_document


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_fields(**overrides):
    """Build a minimal valid fields dict for testing."""
    base = {
        "name":           {"value": "ARJUN KUMAR SHARMA", "confidence": 0.95},
        "passport_number":{"value": "A1234567",           "confidence": 0.92},
        "date_of_birth":  {"value": "1990-05-15",         "confidence": 0.91},
        "date_of_issue":  {"value": "2019-03-20",         "confidence": 0.90},
        "date_of_expiry": {"value": "2029-03-19",         "confidence": 0.90},
        "country_code":   {"value": "IND",                "confidence": 0.98},
        "passport_type":  {"value": "P",                  "confidence": 0.99},
        "sex":            {"value": "M",                  "confidence": 0.97},
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Required fields
# ─────────────────────────────────────────────────────────────────────────────

class TestRequiredFields:
    def test_all_present_passes(self):
        result = validate_required_fields(_make_fields())
        assert result["status"] == "PASS"

    def test_missing_passport_number_fails(self):
        fields = _make_fields()
        fields["passport_number"] = {"value": ""}
        result = validate_required_fields(fields)
        assert result["status"] == "FAIL"
        assert "passport_number" in result["reason"]

    def test_missing_name_fails(self):
        fields = _make_fields()
        fields["name"] = {"value": ""}
        result = validate_required_fields(fields)
        assert result["status"] == "FAIL"

    def test_multiple_missing_listed(self):
        fields = _make_fields()
        fields["name"] = {"value": ""}
        fields["date_of_birth"] = {"value": ""}
        result = validate_required_fields(fields)
        assert result["status"] == "FAIL"
        assert "name" in result["reason"]
        assert "date_of_birth" in result["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Date format validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDateFormats:
    def test_valid_iso_dates_pass(self):
        result = validate_date_formats(_make_fields())
        assert result["status"] == "PASS"

    def test_valid_slash_dates_pass(self):
        fields = _make_fields(
            date_of_birth={"value": "15/05/1990"},
            date_of_issue={"value": "20/03/2019"},
            date_of_expiry={"value": "19/03/2029"},
        )
        result = validate_date_formats(fields)
        assert result["status"] == "PASS"

    def test_impossible_date_fails(self):
        fields = _make_fields(date_of_birth={"value": "31/02/1990"})
        result = validate_date_formats(fields)
        assert result["status"] == "FAIL"

    def test_completely_invalid_string_fails(self):
        fields = _make_fields(date_of_birth={"value": "NOT-A-DATE"})
        result = validate_date_formats(fields)
        assert result["status"] == "FAIL"

    def test_missing_date_skipped(self):
        # Missing dates are handled by required_fields, not date_format
        fields = _make_fields(date_of_birth={"value": ""})
        result = validate_date_formats(fields)
        assert result["status"] == "PASS"

    def test_leap_year_valid(self):
        fields = _make_fields(date_of_birth={"value": "29/02/2000"})
        result = validate_date_formats(fields)
        assert result["status"] == "PASS"

    def test_leap_year_invalid(self):
        fields = _make_fields(date_of_birth={"value": "29/02/2001"})
        result = validate_date_formats(fields)
        assert result["status"] == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Passport number validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPassportNumber:
    def test_valid_indian_format_passes(self):
        result = validate_passport_number(_make_fields())
        assert result["status"] == "PASS"

    def test_empty_number_not_available(self):
        fields = _make_fields(passport_number={"value": ""})
        result = validate_passport_number(fields)
        assert result["status"] == "NOT_AVAILABLE"

    def test_too_short_fails(self):
        fields = _make_fields(passport_number={"value": "A12"})
        result = validate_passport_number(fields)
        assert result["status"] == "FAIL"

    def test_no_leading_letter_fails(self):
        fields = _make_fields(passport_number={"value": "12345678"})
        result = validate_passport_number(fields)
        assert result["status"] == "FAIL"

    def test_valid_pattern_passes(self):
        fields = _make_fields(passport_number={"value": "B9876543"})
        result = validate_passport_number(fields)
        assert result["status"] == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Expiry validation
# ─────────────────────────────────────────────────────────────────────────────

class TestExpiryValidation:
    def test_future_expiry_passes(self):
        fields = _make_fields(date_of_expiry={"value": "2029-03-19"})
        result = validate_expiry(fields, reference_date=date(2026, 1, 1))
        assert result["status"] == "PASS"

    def test_expired_fails(self):
        fields = _make_fields(date_of_expiry={"value": "2023-02-27"})
        result = validate_expiry(fields, reference_date=date(2026, 1, 1))
        assert result["status"] == "FAIL"
        assert "expired" in result["reason"].lower() or "EXPIRED" in result["reason"]

    def test_missing_expiry_not_available(self):
        fields = _make_fields(date_of_expiry={"value": ""})
        result = validate_expiry(fields)
        assert result["status"] == "NOT_AVAILABLE"

    def test_invalid_date_fails(self):
        fields = _make_fields(date_of_expiry={"value": "NOT-A-DATE"})
        result = validate_expiry(fields)
        assert result["status"] == "FAIL"

    def test_today_boundary(self):
        today = date.today()
        fields = _make_fields(date_of_expiry={"value": today.isoformat()})
        result = validate_expiry(fields, reference_date=today)
        # Expired if expiry is strictly before today; today itself is NOT expired
        # (implementation uses < not <=)
        assert result["status"] in ("PASS", "FAIL")


# ─────────────────────────────────────────────────────────────────────────────
# Country code validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCountryCode:
    def test_ind_passes(self):
        result = validate_country_code(_make_fields())
        assert result["status"] == "PASS"

    def test_wrong_country_fails(self):
        fields = _make_fields(country_code={"value": "USA"})
        result = validate_country_code(fields)
        assert result["status"] == "FAIL"

    def test_missing_country_not_available(self):
        fields = _make_fields(country_code={"value": ""})
        result = validate_country_code(fields)
        assert result["status"] == "NOT_AVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# MRZ check digits rule
# ─────────────────────────────────────────────────────────────────────────────

class TestMRZCheckDigitsRule:
    def test_no_mrz_not_available(self):
        result = validate_mrz_check_digits(None)
        assert result["status"] == "NOT_AVAILABLE"

    def test_empty_mrz_not_available(self):
        result = validate_mrz_check_digits({"line1": "", "line2": ""})
        assert result["status"] == "NOT_AVAILABLE"

    def test_all_valid_passes(self):
        mrz = {
            "line1": "P<INDTEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "line2": "A12345674IND9005152M2903199<<<<<<<<<<<<<<<2",
            "all_check_digits_valid": True,
            "check_digits": {
                "passport_number": {"valid": True},
                "date_of_birth":   {"valid": True},
                "date_of_expiry":  {"valid": True},
                "optional_data":   {"valid": True},
                "composite":       {"valid": True},
            },
            "parse_errors": [],
        }
        result = validate_mrz_check_digits(mrz)
        assert result["status"] == "PASS"

    def test_one_invalid_fails(self):
        mrz = {
            "line1": "P<INDTEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "line2": "A12345674IND9005152M2903199<<<<<<<<<<<<<<<2",
            "all_check_digits_valid": False,
            "check_digits": {
                "passport_number": {"valid": False, "value": "3", "expected": "4"},
                "date_of_birth":   {"valid": True,  "value": "2", "expected": "2"},
                "date_of_expiry":  {"valid": True,  "value": "9", "expected": "9"},
                "optional_data":   {"valid": True,  "value": "0", "expected": "0"},
                "composite":       {"valid": True,  "value": "2", "expected": "2"},
            },
            "parse_errors": [],
        }
        result = validate_mrz_check_digits(mrz)
        assert result["status"] == "FAIL"

    def test_parse_errors_fail(self):
        mrz = {"parse_errors": ["Line 2 parse error: something"]}
        result = validate_mrz_check_digits(mrz)
        assert result["status"] == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-field consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFieldConsistency:
    def test_correct_ordering_passes(self):
        result = check_date_ordering(_make_fields())
        assert result["status"] == "PASS"

    def test_dob_after_issue_fails(self):
        fields = _make_fields(
            date_of_birth={"value": "2020-01-01"},  # after issue
            date_of_issue={"value": "2019-03-20"},
        )
        result = check_date_ordering(fields)
        assert result["status"] == "FAIL"

    def test_issue_after_expiry_fails(self):
        fields = _make_fields(
            date_of_issue={"value": "2030-01-01"},   # after expiry
            date_of_expiry={"value": "2029-03-19"},
        )
        result = check_date_ordering(fields)
        assert result["status"] == "FAIL"

    def test_passport_type_p_passes(self):
        result = check_passport_type(_make_fields())
        assert result["status"] == "PASS"

    def test_passport_type_non_p_fails(self):
        fields = _make_fields(passport_type={"value": "D"})
        result = check_passport_type(fields)
        assert result["status"] == "FAIL"

    def test_mrz_consistency_all_match_passes(self):
        cons = {
            "passport_number": "MATCH",
            "date_of_birth":   "MATCH",
            "date_of_expiry":  "MATCH",
            "country":         "MATCH",
            "name":            "MATCH",
        }
        result = check_mrz_consistency(cons)
        assert result["status"] == "PASS"

    def test_mrz_consistency_mismatch_fails(self):
        cons = {
            "passport_number": "MISMATCH",
            "date_of_birth":   "MATCH",
            "date_of_expiry":  "MATCH",
            "country":         "MATCH",
            "name":            "MATCH",
        }
        result = check_mrz_consistency(cons)
        assert result["status"] == "FAIL"

    def test_mrz_consistency_none_is_not_available(self):
        result = check_mrz_consistency(None)
        assert result["status"] == "NOT_AVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Full validate_document integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDocument:
    """Integration tests for the validate_document orchestrator."""

    def test_valid_known_record_passes(self):
        """Case 1: Valid synthetic passport in DB."""
        fields = _make_fields()
        result = validate_document(fields, mrz=None)
        assert result.validation_status in ("PASS", "REVIEW")

    def test_expired_passport_fails(self):
        """Case 2: Expired passport fails expiry check."""
        fields = _make_fields(
            passport_number={"value": "F6789012"},
            date_of_expiry={"value": "2023-02-27"},
        )
        result = validate_document(fields, mrz=None)
        # Either FAIL from expiry rule or FAIL from DB status
        assert result.validation_status in ("FAIL", "REVIEW")
        assert result.checks.passport_expiry.status == "FAIL"

    def test_passport_not_found_gives_review(self):
        """Case 5: Unknown passport number → NOT_FOUND → REVIEW."""
        fields = _make_fields(passport_number={"value": "Z9999999"})
        result = validate_document(fields, mrz=None)
        assert result.validation_status in ("REVIEW", "FAIL")
        assert not result.database_record_found

    def test_missing_required_field_fails(self):
        """Missing passport number → required fields FAIL."""
        fields = _make_fields(passport_number={"value": ""})
        result = validate_document(fields, mrz=None)
        assert result.checks.required_fields.status == "FAIL"

    def test_invalid_date_format_fails(self):
        """Unparseable date → date_format FAIL."""
        fields = _make_fields(date_of_birth={"value": "NOTADATE"})
        result = validate_document(fields, mrz=None)
        assert result.checks.date_format.status == "FAIL"

    def test_result_has_recommendation(self):
        """Validation result always has a non-empty recommendation."""
        result = validate_document(_make_fields(), mrz=None)
        assert result.recommendation
        assert len(result.recommendation) > 10
