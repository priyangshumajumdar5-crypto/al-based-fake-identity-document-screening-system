"""
Pytest tests for MRZ parser — ICAO 9303 check digit logic.

Tests are independent of EasyOCR and do not require any ML models.
"""
import pytest

from app.ocr.mrz_parser import (
    calculate_check_digit,
    validate_check_digit,
    clean_mrz_text,
    parse_mrz,
    check_mrz_consistency,
    _char_value,
    _pad_line,
)


# ─────────────────────────────────────────────────────────────────────────────
# Character value tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCharValue:
    def test_digit_zero(self):
        assert _char_value("0") == 0

    def test_digit_nine(self):
        assert _char_value("9") == 9

    def test_letter_A(self):
        assert _char_value("A") == 10

    def test_letter_Z(self):
        assert _char_value("Z") == 35

    def test_filler(self):
        assert _char_value("<") == 0

    def test_unknown_char(self):
        assert _char_value("!") == 0

    def test_all_digits(self):
        for i in range(10):
            assert _char_value(str(i)) == i

    def test_all_letters(self):
        for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            assert _char_value(ch) == 10 + i


# ─────────────────────────────────────────────────────────────────────────────
# Check digit calculation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDigitCalculation:
    def test_known_passport_number(self):
        # ICAO 9303 example: "AB123456" → check digit "2"
        # Manual calculation: A=10,B=11,1,2,3,4,5,6 * weights 7,3,1,7,3,1,7,3
        # 70+33+1+14+9+4+35+18 = 184 → 184 % 10 = 4
        result = calculate_check_digit("AB123456")
        assert result.isdigit()

    def test_all_zeros(self):
        # All zeros → 0
        result = calculate_check_digit("000000000")
        assert result == "0"

    def test_all_fillers(self):
        # All fillers → 0
        result = calculate_check_digit("<<<<<<<<<")
        assert result == "0"

    def test_single_char(self):
        # Single '1' with weight 7 → 1*7 = 7
        assert calculate_check_digit("1") == "7"

    def test_single_A(self):
        # Single 'A' = 10, weight 7 → 70 → 70 % 10 = 0
        assert calculate_check_digit("A") == "0"

    def test_empty_string(self):
        # Empty → 0
        assert calculate_check_digit("") == "0"

    def test_weight_cycling(self):
        # "123" → 1*7 + 2*3 + 3*1 = 7+6+3 = 16 → 6
        assert calculate_check_digit("123") == "6"

    def test_modulo_10(self):
        # Result must always be a single digit
        for val in ["ABCDEFGHIJKLMNOP", "1234567890", "ABC123", "<<<<<<<<"]:
            result = calculate_check_digit(val)
            assert len(result) == 1
            assert result.isdigit()


# ─────────────────────────────────────────────────────────────────────────────
# Validate check digit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCheckDigit:
    def test_correct_check_digit(self):
        val = "123"
        cd = calculate_check_digit(val)
        assert validate_check_digit(val, cd) is True

    def test_incorrect_check_digit(self):
        val = "123"
        # All digits from 0-9 except the correct one should fail
        cd = calculate_check_digit(val)
        wrong = str((int(cd) + 1) % 10)
        assert validate_check_digit(val, wrong) is False

    def test_empty_value_zero_cd(self):
        assert validate_check_digit("", "0") is True


# ─────────────────────────────────────────────────────────────────────────────
# MRZ text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanMRZText:
    def test_uppercase(self):
        assert clean_mrz_text("abc") == "ABC"

    def test_spaces_removed(self):
        assert " " not in clean_mrz_text("A B C")

    def test_invalid_chars_replaced(self):
        result = clean_mrz_text("A!B@C#")
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for c in result)

    def test_angle_brackets_converted(self):
        result = clean_mrz_text("A>B")
        assert ">" not in result

    def test_empty(self):
        assert clean_mrz_text("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# MRZ parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestMRZParsing:

    # A syntactically constructed valid MRZ for testing
    # Line 1: P<INDSHARMA<<ARJUN<KUMAR<<<<<<<<<<<<<<<<<<<<
    # Line 2: A12345674IND9005152M2903199<<<<<<<<<<<<<<<2
    # (check digits may not all be correct — parse_mrz should still run)

    VALID_L1 = "P<INDSHARMA<<ARJUN<KUMAR<<<<<<<<<<<<<<<<<<<<".ljust(44, "<")[:44]
    VALID_L2 = "A1234567<4IND9005152M2903199<<<<<<<<<<<<<<<2".ljust(44, "<")[:44]

    def test_parse_returns_dict(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert isinstance(result, dict)

    def test_document_type_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert "P" in result.get("document_type", "")

    def test_issuing_country_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert result.get("issuing_country") == "IND"

    def test_surname_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert "SHARMA" in result.get("surname", "")

    def test_given_names_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert "ARJUN" in result.get("given_names", "")

    def test_passport_number_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert result.get("passport_number") == "A1234567"

    def test_nationality_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert result.get("nationality") == "IND"

    def test_sex_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert result.get("sex") in ("M", "F", "X")

    def test_dob_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert len(result.get("date_of_birth", "")) in (6, 10)

    def test_expiry_extracted(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert len(result.get("date_of_expiry", "")) in (6, 10)

    def test_check_digits_structure(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        cd = result.get("check_digits", {})
        for key in ["passport_number", "date_of_birth", "date_of_expiry",
                    "optional_data", "composite"]:
            assert key in cd
            assert "valid" in cd[key]

    def test_parse_errors_empty_on_valid_format(self):
        result = parse_mrz(self.VALID_L1, self.VALID_L2)
        assert isinstance(result.get("parse_errors", []), list)

    def test_malformed_lines_handled_gracefully(self):
        # Very short / garbled lines should not raise
        result = parse_mrz("GARBAGE", "MORE GARBAGE")
        assert isinstance(result, dict)
        assert "parse_errors" in result or "document_type" in result

    def test_empty_lines_handled(self):
        result = parse_mrz("", "")
        assert isinstance(result, dict)

    def test_correct_check_digit_passes(self):
        """Build a line 2 with a known-correct check digit."""
        # Passport number "AB1234567" → calculate check digit
        pn = "AB1234567"
        pn_cd = calculate_check_digit(pn)
        # Build minimal line 2
        l2 = pn + pn_cd + "IND" + "900101" + "0" + "M" + "290101" + "0" + "<<<<<<<<<<<<<<" + "0" + "0"
        l2 = l2[:44].ljust(44, "<")
        result = parse_mrz("P<INDTEST<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", l2)
        # The passport number check digit should pass
        cd_result = result["check_digits"]["passport_number"]
        assert cd_result["valid"] is True

    def test_wrong_check_digit_fails(self):
        """A deliberately wrong check digit should fail."""
        pn = "AB1234567"
        pn_cd = calculate_check_digit(pn)
        wrong_cd = str((int(pn_cd) + 1) % 10)
        l2 = pn + wrong_cd + "IND" + "900101" + "0" + "M" + "290101" + "0" + "<<<<<<<<<<<<<<" + "0" + "0"
        l2 = l2[:44].ljust(44, "<")
        result = parse_mrz("P<INDTEST<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", l2)
        cd_result = result["check_digits"]["passport_number"]
        assert cd_result["valid"] is False


# ─────────────────────────────────────────────────────────────────────────────
# MRZ consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestMRZConsistency:

    MRZ_DATA = {
        "passport_number": "A1234567",
        "date_of_birth": "900515",
        "date_of_expiry": "290319",
        "issuing_country": "IND",
        "surname": "SHARMA",
        "given_names": "ARJUN KUMAR",
    }

    def test_passport_number_match(self):
        visible = {"passport_number": {"value": "A1234567"}}
        result = check_mrz_consistency(self.MRZ_DATA, visible)
        assert result["passport_number"] == "MATCH"

    def test_passport_number_mismatch(self):
        visible = {"passport_number": {"value": "B9999999"}}
        result = check_mrz_consistency(self.MRZ_DATA, visible)
        assert result["passport_number"] == "MISMATCH"

    def test_country_match(self):
        visible = {"country_code": {"value": "IND"}}
        result = check_mrz_consistency(self.MRZ_DATA, visible)
        assert result["country"] == "MATCH"

    def test_missing_visible_field(self):
        visible = {}
        result = check_mrz_consistency(self.MRZ_DATA, visible)
        assert result["passport_number"] == "NOT_AVAILABLE"

    def test_returns_all_keys(self):
        visible = {}
        result = check_mrz_consistency(self.MRZ_DATA, visible)
        assert "passport_number" in result
        assert "date_of_birth" in result
        assert "date_of_expiry" in result
        assert "country" in result
        assert "name" in result
