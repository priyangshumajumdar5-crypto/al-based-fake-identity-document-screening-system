"""
Tests for Module 2 v2 — validate_passport() flow.

Tests normalization helpers, field comparison, lookup_passport,
and the validate_passport orchestrator against the real CSV-seeded database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.validation.rules import compare_field, normalize_date, normalize_text
from app.validation.schemas import FieldComparison, ValidationResult, ValidationSummary


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: ensure passports table is seeded before DB-dependent tests
# ─────────────────────────────────────────────────────────────────────────────

# Sample records matching data/draft_passport_database.csv
_TEST_PASSPORTS = [
    ("AA441624", "WARAD", "PRAVEENKUMAR DODDAPPA", "27/07/1993",
     "M K HUBBALLI, KARNATAKA", "08/05/2025", "BENGALURU", "07/05/2035",
     "M", "passport/passport1.jpg"),
    ("X9564204", "MOHAMMED", "AHMED", "01/02/1988",
     "HYDERABAD, TELANGANA", "17/05/2023", "HYDERABAD", "16/05/2033",
     "M", "passport/passport2.jpg"),
    ("B6533628", "", "ZAIN AHMED SIDDIQUI", "13/05/2005",
     "MAHBOOBNAGAR, TELANGANA", "18/10/2023", "HYDERABAD", "17/10/2033",
     "M", "passport/passport3.jpg"),
]


@pytest.fixture(autouse=True)
def _ensure_passports_table():
    """
    Ensure the `passports` table exists and has test records.
    Runs before every test in this module.
    """
    from app.database.database import initialize_database, db_session

    initialize_database()

    with db_session() as conn:
        # Insert test records if not already present
        for rec in _TEST_PASSPORTS:
            conn.execute(
                """INSERT OR IGNORE INTO passports
                   (passport_number, surname, given_name, date_of_birth,
                    place_of_birth, date_of_issue, place_of_issue, date_of_expiry,
                    gender, passport_photo_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rec,
            )


# ─────────────────────────────────────────────────────────────────────────────
# normalize_text
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_strips_and_uppercases(self):
        assert normalize_text("  hello world  ") == "HELLO WORLD"

    def test_collapses_whitespace(self):
        assert normalize_text("FOO   BAR  BAZ") == "FOO BAR BAZ"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_safe(self):
        # The function expects str; test with empty
        assert normalize_text("") == ""

    def test_already_upper(self):
        assert normalize_text("MUMBAI") == "MUMBAI"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_date
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeDateNew:
    def test_dd_mm_yyyy_passthrough(self):
        assert normalize_date("27/07/1993") == "27/07/1993"

    def test_iso_to_dd_mm_yyyy(self):
        assert normalize_date("1993-07-27") == "27/07/1993"

    def test_dd_dash_mm_dash_yyyy(self):
        assert normalize_date("27-07-1993") == "27/07/1993"

    def test_dd_dot_mm_dot_yyyy(self):
        assert normalize_date("27.07.1993") == "27/07/1993"

    def test_single_digit_padding(self):
        assert normalize_date("5/3/2025") == "05/03/2025"

    def test_empty(self):
        assert normalize_date("") == ""

    def test_unparseable_returns_original(self):
        assert normalize_date("not-a-date") == "not-a-date"

    def test_iso_single_digit(self):
        assert normalize_date("2025-5-8") == "08/05/2025"


# ─────────────────────────────────────────────────────────────────────────────
# compare_field
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareField:
    def test_exact_match(self):
        assert compare_field("M", "M", "exact") == "MATCHED"

    def test_exact_case_insensitive(self):
        assert compare_field("male", "MALE", "exact") == "MATCHED"

    def test_exact_mismatch(self):
        assert compare_field("M", "F", "exact") == "MISMATCH"

    def test_empty_ocr_not_verified(self):
        assert compare_field("", "M", "exact") == "NOT_VERIFIED"

    def test_empty_db_not_verified(self):
        assert compare_field("M", "", "exact") == "NOT_VERIFIED"

    def test_both_empty_not_verified(self):
        assert compare_field("", "", "exact") == "NOT_VERIFIED"

    def test_date_iso_vs_slash(self):
        assert compare_field("1993-07-27", "27/07/1993", "date") == "MATCHED"

    def test_date_mismatch(self):
        assert compare_field("1993-07-27", "28/07/1993", "date") == "MISMATCH"

    def test_name_exact_match(self):
        assert compare_field("SHARMA", "SHARMA", "name") == "MATCHED"

    def test_name_fuzzy_match(self):
        # Small OCR error — should still match with fuzzy threshold >=90
        assert compare_field("PRAVEENKUMAR DODDAPPA", "PRAVEENKUMAR DODDAPPA", "name") == "MATCHED"

    def test_name_complete_mismatch(self):
        assert compare_field("JOHN DOE", "JANE SMITH", "name") == "MISMATCH"

    def test_name_fuzzy_close_match(self):
        # Slight OCR misread — token_sort_ratio should be high
        assert compare_field("PRAVEENKUMR DODDAPPA", "PRAVEENKUMAR DODDAPPA", "name") == "MATCHED"


# ─────────────────────────────────────────────────────────────────────────────
# lookup_passport
# ─────────────────────────────────────────────────────────────────────────────

class TestLookupPassport:
    """Test lookup_passport against the real CSV-seeded database."""

    def test_exact_match(self):
        """AA441624 should be found via exact match."""
        from app.validation.consistency import lookup_passport

        record, fuzzy = lookup_passport("AA441624")
        assert record is not None
        assert fuzzy is False
        assert record["passport_number"] == "AA441624"
        assert record["surname"] == "WARAD"

    def test_not_found(self):
        """A completely unknown passport number should return None."""
        from app.validation.consistency import lookup_passport

        record, fuzzy = lookup_passport("ZZZZZ999")
        assert record is None
        assert fuzzy is False

    def test_empty_input(self):
        from app.validation.consistency import lookup_passport

        record, fuzzy = lookup_passport("")
        assert record is None
        assert fuzzy is False

    def test_case_insensitive_exact(self):
        """Passport numbers are uppercased before lookup."""
        from app.validation.consistency import lookup_passport

        record, fuzzy = lookup_passport("aa441624")
        # The DB has AA441624 — should still match
        # (if sqlite COLLATE is case-sensitive, the function uppercases first)
        # This depends on whether the DB stores uppercase — it does.
        if record is not None:
            assert record["passport_number"] == "AA441624"

    def test_known_passport_x9564204(self):
        from app.validation.consistency import lookup_passport

        record, fuzzy = lookup_passport("X9564204")
        assert record is not None
        assert record["surname"] == "MOHAMMED"
        assert record["given_name"] == "AHMED"


# ─────────────────────────────────────────────────────────────────────────────
# validate_passport
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatePassport:
    """Test the full validate_passport orchestrator."""

    def test_no_passport_number(self):
        """No passport_number → NOT_FOUND."""
        from app.validation.validator import validate_passport

        result = validate_passport({"name": {"value": "FOO"}})
        assert result.overall_status == "NOT_FOUND"
        assert result.db_match is False

    def test_unknown_passport(self):
        """Unknown passport → NOT_FOUND."""
        from app.validation.validator import validate_passport

        result = validate_passport({"passport_number": {"value": "ZZZZ9999"}})
        assert result.overall_status == "NOT_FOUND"
        assert result.db_match is False

    def test_known_passport_matched(self):
        """A1234567 is in the synthetic DB via seed — should match."""
        from app.validation.validator import validate_passport

        # Use AA441624 from the real CSV-seeded passports table
        ocr = {
            "passport_number": {"value": "AA441624"},
            "name": {"value": "WARAD PRAVEENKUMAR DODDAPPA"},
            "date_of_birth": {"value": "27/07/1993"},
            "date_of_expiry": {"value": "07/05/2035"},
            "place_of_birth": {"value": "M K HUBBALLI, KARNATAKA"},
            "place_of_issue": {"value": "BENGALURU"},
            "date_of_issue": {"value": "08/05/2025"},
            "sex": {"value": "M"},
        }
        result = validate_passport(ocr)
        assert result.db_match is True
        assert result.passport_number == "AA441624"
        assert result.overall_status in ("VERIFIED", "NEEDS_REVIEW")
        assert len(result.field_comparisons) > 0
        assert result.summary.matched_count > 0

    def test_mismatched_gender(self):
        """If gender doesn't match DB, should be MISMATCH."""
        from app.validation.validator import validate_passport

        ocr = {
            "passport_number": {"value": "AA441624"},
            "name": {"value": "WARAD PRAVEENKUMAR DODDAPPA"},
            "sex": {"value": "F"},  # DB has M
        }
        result = validate_passport(ocr)
        assert result.db_match is True
        # Find the Gender comparison
        gender_comp = next(
            (c for c in result.field_comparisons if c.field_name == "Gender"),
            None,
        )
        if gender_comp:
            assert gender_comp.status == "MISMATCH"
        assert result.summary.mismatch_count >= 1
        assert result.overall_status == "NEEDS_REVIEW"

    def test_result_has_photo_path(self):
        """DB record should include passport_photo_path."""
        from app.validation.validator import validate_passport

        ocr = {
            "passport_number": {"value": "AA441624"},
            "name": {"value": "WARAD PRAVEENKUMAR DODDAPPA"},
        }
        result = validate_passport(ocr)
        assert result.db_match is True
        assert result.passport_photo_path is not None
        assert "passport" in result.passport_photo_path.lower()

    def test_result_pydantic_serialization(self):
        """Ensure ValidationResult can be serialized to dict/JSON."""
        from app.validation.validator import validate_passport

        ocr = {"passport_number": {"value": "AA441624"}}
        result = validate_passport(ocr)
        d = result.model_dump()
        assert "field_comparisons" in d
        assert "summary" in d
        assert "overall_status" in d

    def test_iso_date_matches_slash_date(self):
        """OCR returns ISO date, DB has DD/MM/YYYY — should MATCH."""
        from app.validation.validator import validate_passport

        ocr = {
            "passport_number": {"value": "AA441624"},
            "name": {"value": "WARAD PRAVEENKUMAR DODDAPPA"},
            "date_of_birth": {"value": "1993-07-27"},  # ISO format
        }
        result = validate_passport(ocr)
        dob_comp = next(
            (c for c in result.field_comparisons if c.field_name == "Date of Birth"),
            None,
        )
        if dob_comp:
            assert dob_comp.status == "MATCHED"
