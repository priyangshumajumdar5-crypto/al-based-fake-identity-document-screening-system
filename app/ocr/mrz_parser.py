"""
ICAO Doc 9303 TD3 MRZ Parser for Indian Passports.

Identifies, cleans, parses, and validates MRZ text and check digits.
Converts MRZ date strings (YYMMDD) to DD/MM/YYYY format.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _char_value(ch: str) -> int:
    """ICAO 9303 character numeric value mapping."""
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    return 0


def calculate_check_digit(value: str) -> str:
    """
    Calculate ICAO 9303 check digit using [7, 3, 1] weights.

    Args:
        value: Alphanumeric string.

    Returns:
        Check digit as string ("0"-"9").
    """
    weights = [7, 3, 1]
    total = sum(_char_value(c) * weights[i % 3] for i, c in enumerate(value))
    return str(total % 10)


def validate_check_digit(value: str, expected: str) -> bool:
    """Verify if check digit matches expected value."""
    return calculate_check_digit(value) == expected


def parse_mrz_date(yymmdd: str, field_type: str) -> str:
    """
    Convert MRZ YYMMDD date string to 'DD/MM/YYYY' format.

    Args:
        yymmdd: 6-digit MRZ date string.
        field_type: Either 'birth' or 'expiry'.

    Returns:
        Formatted date string 'DD/MM/YYYY' or original if invalid.
    """
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd

    yy = int(yymmdd[0:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]

    current_yy = datetime.now().year % 100

    if field_type == "expiry":
        year = 2000 + yy
    else:  # birth
        year = (1900 + yy) if yy > current_yy else (2000 + yy)

    return f"{dd}/{mm}/{year}"


CHAR_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1', 'l': '1', '|': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8',
    'g': '9', 'q': '9',
}


def _fix_digits(s: str) -> str:
    """Convert common OCR misread letters to digits in numeric MRZ fields."""
    res = []
    for c in s:
        if c.isdigit():
            res.append(c)
        elif c in CHAR_TO_DIGIT:
            res.append(CHAR_TO_DIGIT[c])
        else:
            res.append(c)
    return "".join(res)


def clean_mrz_text(raw: str) -> str:
    """Clean OCR text into valid MRZ character sequence."""
    if not raw:
        return ""
    cleaned = raw.upper().replace(" ", "")
    cleaned = cleaned.replace("«", "<").replace("»", "<").replace(">", "<").replace("|", "<")
    cleaned = re.sub(r"[^A-Z0-9<]", "<", cleaned)
    return cleaned


def _pad_line(line: str, length: int = 44) -> str:
    """Pad or truncate MRZ line to target length."""
    line = line[:length]
    return line.ljust(length, "<")


def clean_mrz_line(line: str, line_length: int = 44) -> str:
    """Clean and pad/truncate MRZ line to target length."""
    cleaned = clean_mrz_text(line)
    # Fix common misread filler K in MRZ lines where K is preceded or followed by <
    cleaned = re.sub(r"(?<=[<A-Z0-9])K+(?=[<])|(?<=[<])K+(?=[<A-Z0-9])", lambda m: "<" * len(m.group(0)), cleaned)
    return _pad_line(cleaned, line_length)


def identify_mrz_lines(ocr_results: List[Any]) -> Tuple[str, str]:
    """
    Extract MRZ lines from raw OCR detections or list of text strings.

    Args:
        ocr_results: List of (bbox, text, conf) tuples or list of text strings.

    Returns:
        (line1, line2) tuple of ~44-character MRZ strings.
    """
    texts: List[str] = []
    for item in ocr_results:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            texts.append(str(item[1]))
        elif isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
        elif isinstance(item, str):
            texts.append(item)

    candidates: List[str] = []
    for t in texts:
        cleaned = clean_mrz_text(t)
        if len(cleaned) >= 28 or "<" in cleaned:
            candidates.append(cleaned)

    if len(candidates) >= 2:
        return clean_mrz_line(candidates[-2]), clean_mrz_line(candidates[-1])
    elif len(candidates) == 1:
        return clean_mrz_line(candidates[0]), ""
    return "", ""


def parse_mrz(line1: str, line2: str) -> Dict[str, Any]:
    """
    Parse TD3 MRZ lines into structured fields and check-digit validation.

    Args:
        line1: 44-char first MRZ line.
        line2: 44-char second MRZ line.

    Returns:
        Dictionary containing extracted fields and mrz_validation dict.
    """
    l1 = clean_mrz_line(line1)
    l2 = clean_mrz_line(line2)

    doc_type = l1[0:2].replace("<", "").strip()
    country_code = l1[2:5].replace("<", "").strip()

    name_field = l1[5:44]
    # Clean filler noise K from names
    name_field = re.sub(r"\bK+\b", "", name_field)
    if "<<" in name_field:
        parts = name_field.split("<<", 1)
        surname = parts[0].replace("<", " ").strip()
        given_names = parts[1].replace("<", " ").strip()
    else:
        surname = name_field.replace("<", " ").strip()
        given_names = ""

    pn_field = l2[0:9]
    pn_check = _fix_digits(l2[9])
    nationality = l2[10:13].replace("<", "").strip()
    dob_field = _fix_digits(l2[13:19])
    dob_check = _fix_digits(l2[19])
    sex = l2[20] if l2[20] in ("M", "F") else "X"
    exp_field = _fix_digits(l2[21:27])
    exp_check = _fix_digits(l2[27])
    opt_field = l2[28:42]
    opt_check = _fix_digits(l2[42])
    composite_check = _fix_digits(l2[43])

    passport_number = pn_field.replace("<", "").strip()
    personal_number = opt_field.replace("<", "").strip()

    # Checksum validation calculations (using raw 6-digit MRZ fields)
    pn_calc = calculate_check_digit(pn_field)
    pn_valid = pn_check == pn_calc
    if not pn_valid:
        print(f"[MRZ CHECKSUM FAIL] Field 'passport_number': expected check digit '{pn_check}' vs computed '{pn_calc}' vs raw substring '{pn_field}'")

    dob_calc = calculate_check_digit(dob_field)
    dob_valid = dob_check == dob_calc
    if not dob_valid:
        print(f"[MRZ CHECKSUM FAIL] Field 'date_of_birth': expected check digit '{dob_check}' vs computed '{dob_calc}' vs raw substring '{dob_field}'")

    exp_calc = calculate_check_digit(exp_field)
    exp_valid = exp_check == exp_calc
    if not exp_valid:
        print(f"[MRZ CHECKSUM FAIL] Field 'date_of_expiry': expected check digit '{exp_check}' vs computed '{exp_calc}' vs raw substring '{exp_field}'")

    opt_calc = calculate_check_digit(opt_field)
    opt_valid = opt_check == opt_calc
    if not opt_valid:
        print(f"[MRZ CHECKSUM FAIL] Field 'personal_number': expected check digit '{opt_check}' vs computed '{opt_calc}' vs raw substring '{opt_field}'")

    composite_str = l2[0:10] + l2[13:20] + l2[21:43]
    comp_calc = calculate_check_digit(composite_str)
    comp_valid = composite_check == comp_calc
    if not comp_valid:
        print(f"[MRZ CHECKSUM FAIL] Field 'composite': expected check digit '{composite_check}' vs computed '{comp_calc}' vs raw substring '{composite_str}'")

    # Format dates into DD/MM/YYYY
    formatted_dob = parse_mrz_date(dob_field, "birth")
    formatted_exp = parse_mrz_date(exp_field, "expiry")

    mrz_validation = {
        "passport_number": "PASS" if pn_valid else "FAIL",
        "date_of_birth": "PASS" if dob_valid else "FAIL",
        "date_of_expiry": "PASS" if exp_valid else "FAIL",
        "personal_number": "PASS" if opt_valid else "FAIL",
        "composite": "PASS" if comp_valid else "FAIL",
    }

    check_digits = {
        "passport_number": {"value": pn_check, "expected": pn_calc, "valid": pn_valid},
        "date_of_birth": {"value": dob_check, "expected": dob_calc, "valid": dob_valid},
        "date_of_expiry": {"value": exp_check, "expected": exp_calc, "valid": exp_valid},
        "optional_data": {"value": opt_check, "expected": opt_calc, "valid": opt_valid},
        "composite": {"value": composite_check, "expected": comp_calc, "valid": comp_valid},
    }

    return {
        "line1": l1,
        "line2": l2,
        "document_type": doc_type,
        "country_code": country_code,
        "issuing_country": country_code,
        "surname": surname,
        "given_names": given_names,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": formatted_dob,
        "sex": sex,
        "date_of_expiry": formatted_exp,
        "personal_number": personal_number,
        "optional_data": personal_number,
        "mrz_validation": mrz_validation,
        "check_digits": check_digits,
        "all_check_digits_valid": all([pn_valid, dob_valid, exp_valid, comp_valid]),
        "parse_errors": [],
    }


def check_mrz_consistency(mrz: Dict[str, Any], visible_fields: Dict[str, Any]) -> Dict[str, str]:
    """Compare MRZ fields with visible fields."""
    def _vis(k: str) -> str:
        v = visible_fields.get(k, "")
        if isinstance(v, dict):
            return str(v.get("value", ""))
        return str(v)

    def _comp(a: str, b: str) -> str:
        if not a or not b:
            return "NOT_AVAILABLE"
        return "MATCH" if a.strip().upper() == b.strip().upper() else "MISMATCH"

    mrz_pn = mrz.get("passport_number", "").replace("<", "").strip()
    vis_pn = _vis("passport_number")

    mrz_dob = mrz.get("date_of_birth", "")
    vis_dob_digits = re.sub(r"\D", "", _vis("date_of_birth"))
    mrz_dob_digits = re.sub(r"\D", "", mrz_dob)

    mrz_exp = mrz.get("date_of_expiry", "")
    vis_exp_digits = re.sub(r"\D", "", _vis("date_of_expiry"))
    mrz_exp_digits = re.sub(r"\D", "", mrz_exp)

    mrz_country = mrz.get("issuing_country", mrz.get("country_code", "")).replace("<", "").strip()
    vis_country = _vis("country_code")

    mrz_surname = mrz.get("surname", "").upper().strip()
    mrz_given = mrz.get("given_names", "").upper().strip()
    mrz_full = f"{mrz_surname} {mrz_given}".strip()
    vis_name = _vis("name").upper().strip()

    return {
        "passport_number": _comp(mrz_pn, vis_pn),
        "date_of_birth": _comp(mrz_dob_digits[-6:], vis_dob_digits[-6:]) if mrz_dob_digits and vis_dob_digits else "NOT_AVAILABLE",
        "date_of_expiry": _comp(mrz_exp_digits[-6:], vis_exp_digits[-6:]) if mrz_exp_digits and vis_exp_digits else "NOT_AVAILABLE",
        "country": _comp(mrz_country, vis_country),
        "name": _comp(mrz_full, vis_name),
    }
