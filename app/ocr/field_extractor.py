"""
Field extractor for visual zone fields (place_of_birth, place_of_issue, date_of_issue)
using independent fuzzy label matching and positional text ordering.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Tuple

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b"
)


def _get_y_center(block: Dict[str, Any]) -> float:
    """Calculate vertical center of an OCR text box."""
    bbox = block.get("bbox")
    if not bbox or not isinstance(bbox, (list, tuple)):
        return 0.0
    ys = [pt[1] for pt in bbox if isinstance(pt, (list, tuple)) and len(pt) >= 2]
    return sum(ys) / len(ys) if ys else 0.0


def clean_text_value(text: str) -> str:
    """Clean extracted value text by removing non-alphanumeric noise."""
    text = text.strip()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text.strip()


def is_noise(text: str) -> bool:
    """Return True if text is purely noise (stray symbols, length < 3)."""
    cleaned = clean_text_value(text)
    if len(cleaned) < 3:
        return True
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return True
    return False


def format_date_str(raw_date: str) -> str:
    """Format raw date string to DD/MM/YYYY."""
    m = DATE_PATTERN.search(raw_date)
    if not m:
        return raw_date
    val = m.group(1).replace("-", "/").replace(".", "/")
    parts = val.split("/")
    if len(parts) == 3:
        p1, p2, p3 = parts[0], parts[1], parts[2]
        if len(p1) == 4:  # YYYY/MM/DD
            return f"{p3.zfill(2)}/{p2.zfill(2)}/{p1}"
        elif len(p3) == 2:  # DD/MM/YY
            year = f"20{p3}"
            return f"{p1.zfill(2)}/{p2.zfill(2)}/{year}"
        elif len(p3) == 4:
            return f"{p1.zfill(2)}/{p2.zfill(2)}/{p3}"
    return val


def fuzzy_score_phrase(ocr_line: str, target_phrase: str) -> float:
    """Calculate sequence matcher ratio for full label phrase."""
    clean_line = re.sub(r"[^A-Z\s]", "", ocr_line.upper()).strip()
    clean_target = target_phrase.upper().strip()
    if not clean_line:
        return 0.0
    if clean_target in clean_line:
        return 1.0
    ratio = difflib.SequenceMatcher(None, clean_line, clean_target).ratio()
    return ratio


def extract_visual_zone_fields(ocr_blocks: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, str]:
    """
    Extract place_of_birth, place_of_issue, and date_of_issue independently from visual zone OCR text.

    Args:
        ocr_blocks: List of dicts with 'text', 'bbox', 'confidence'.
        verbose: If True, print match debugging info.

    Returns:
        Dict containing place_of_birth, place_of_issue, date_of_issue.
    """
    sorted_blocks = sorted(ocr_blocks, key=_get_y_center)
    lines = [b.get("text", "").strip() for b in sorted_blocks if b.get("text")]
    n_lines = len(lines)

    extracted = {
        "place_of_birth": "",
        "place_of_issue": "",
        "date_of_issue": "",
    }

    # ── 1. Independent Search for Place of Birth ─────────────────────────────
    birth_matches: List[Tuple[int, float, str]] = []
    for i, line in enumerate(lines):
        score_birth = fuzzy_score_phrase(line, "PLACE OF BIRTH")
        score_issue = fuzzy_score_phrase(line, "PLACE OF ISSUE")
        if score_birth >= 0.50 and score_birth >= score_issue:
            birth_matches.append((i, score_birth, line))

    if birth_matches:
        birth_matches.sort(key=lambda x: x[1], reverse=True)
        idx, score, matched_line = birth_matches[0]
        val = ""
        if ":" in lines[idx] and len(lines[idx].split(":", 1)[1].strip()) >= 3:
            cand = clean_text_value(lines[idx].split(":", 1)[1])
            if not is_noise(cand):
                val = cand
        if not val:
            for offset in (1, 2, 3):
                if idx + offset < n_lines:
                    cand = clean_text_value(lines[idx + offset])
                    if not is_noise(cand) and not DATE_PATTERN.search(cand):
                        val = cand
                        break
        extracted["place_of_birth"] = val
        if verbose:
            print(f"[DEBUG] place_of_birth matched line {idx}: '{matched_line}' (score: {score:.2f}) -> value: '{val}'")

    # ── 2. Independent Search for Place of Issue ─────────────────────────────
    issue_matches: List[Tuple[int, float, str]] = []
    for i, line in enumerate(lines):
        score_issue = fuzzy_score_phrase(line, "PLACE OF ISSUE")
        score_birth = fuzzy_score_phrase(line, "PLACE OF BIRTH")
        if score_issue >= 0.45 and score_issue >= (score_birth - 0.05):
            if birth_matches and i <= birth_matches[0][0] and score_birth >= score_issue:
                continue
            issue_matches.append((i, score_issue, line))

    if issue_matches:
        issue_matches.sort(key=lambda x: x[1], reverse=True)
        idx, score, matched_line = issue_matches[0]
        val = ""
        if ":" in lines[idx] and len(lines[idx].split(":", 1)[1].strip()) >= 3:
            cand = clean_text_value(lines[idx].split(":", 1)[1])
            if not is_noise(cand) and not DATE_PATTERN.search(cand):
                val = cand
        if not val:
            for offset in (1, 2, 3):
                if idx + offset < n_lines:
                    cand = clean_text_value(lines[idx + offset])
                    if not is_noise(cand) and not DATE_PATTERN.search(cand):
                        val = cand
                        break
        extracted["place_of_issue"] = val
        if verbose:
            print(f"[DEBUG] place_of_issue matched line {idx}: '{matched_line}' (score: {score:.2f}) -> value: '{val}'")

    # Layout position fallback if place_of_issue was missed but two "PLACE OF" lines exist
    if not extracted["place_of_issue"] and len(lines) >= 4:
        place_lines = [i for i, l in enumerate(lines) if "PLACE" in l.upper() or "ISSUE" in l.upper() or "ISUT" in l.upper()]
        if len(place_lines) >= 2:
            second_idx = place_lines[-1]
            for offset in (1, 2):
                if second_idx + offset < n_lines:
                    cand = clean_text_value(lines[second_idx + offset])
                    if not is_noise(cand) and not DATE_PATTERN.search(cand):
                        extracted["place_of_issue"] = cand
                        if verbose:
                            print(f"[DEBUG] place_of_issue fallback layout matched line {second_idx}: '{lines[second_idx]}' -> value: '{cand}'")
                        break

    # ── 3. Independent Search for Date of Issue ─────────────────────────────
    date_matches: List[Tuple[int, float, str]] = []
    for i, line in enumerate(lines):
        score_date = fuzzy_score_phrase(line, "DATE OF ISSUE")
        if score_date >= 0.50 or "ISSUE" in line.upper() or "DATE" in line.upper():
            date_matches.append((i, score_date, line))

    date_val = ""
    for idx, score, matched_line in date_matches:
        m = DATE_PATTERN.search(lines[idx])
        if m:
            date_val = format_date_str(m.group(1))
            if verbose:
                print(f"[DEBUG] date_of_issue matched line {idx}: '{matched_line}' (score: {score:.2f}) -> value: '{date_val}'")
            break
        for offset in (1, 2, 3):
            if idx + offset < n_lines:
                m = DATE_PATTERN.search(lines[idx + offset])
                if m:
                    date_val = format_date_str(m.group(1))
                    if verbose:
                        print(f"[DEBUG] date_of_issue matched line {idx + offset}: '{lines[idx + offset]}' (offset: {offset}) -> value: '{date_val}'")
                    break
        if date_val:
            break

    if not date_val:
        all_dates = []
        for line in lines:
            m = DATE_PATTERN.search(line)
            if m:
                all_dates.append(format_date_str(m.group(1)))
        if len(all_dates) >= 3:
            date_val = all_dates[1]
        elif all_dates:
            date_val = all_dates[0]

    extracted["date_of_issue"] = date_val

    return extracted


def merge_mrz_and_visual_fields(
    mrz_data: Dict[str, Any],
    visual_data: Dict[str, str],
    raw_ocr_lines: List[str],
) -> Dict[str, Any]:
    """Merge MRZ parsed fields and visual zone extracted fields into a single dict."""
    surname = mrz_data.get("surname", "")
    given_names = mrz_data.get("given_names", "")

    return {
        "document_type": mrz_data.get("document_type", "P"),
        "country_code": mrz_data.get("country_code", "IND"),
        "surname": surname,
        "given_names": given_names,
        "passport_number": mrz_data.get("passport_number", ""),
        "nationality": mrz_data.get("nationality", "IND"),
        "date_of_birth": mrz_data.get("date_of_birth", ""),
        "sex": mrz_data.get("sex", ""),
        "date_of_expiry": mrz_data.get("date_of_expiry", ""),
        "personal_number": mrz_data.get("personal_number", ""),
        "place_of_birth": visual_data.get("place_of_birth", ""),
        "place_of_issue": visual_data.get("place_of_issue", ""),
        "date_of_issue": visual_data.get("date_of_issue", ""),
        "passport_type": "P",
        "raw_ocr_text": raw_ocr_lines,
        "mrz_validation": mrz_data.get("mrz_validation", {}),
    }


def extract_all_fields(img: Any, use_template: bool = True) -> Dict[str, Any]:
    """
    Extract structured passport fields from an image using EasyOCR + MRZ Parser + Visual Zone Extractor.
    """
    from app.ocr.engine import extract_text_blocks, run_ocr
    from app.ocr.mrz_parser import identify_mrz_lines, parse_mrz
    from app.ocr.preprocessing import preprocess_image

    def _f(v: str, c: float = 0.90) -> Dict[str, Any]:
        return {"value": v or "", "confidence": round(c, 2), "raw": v or ""}

    # 1. Preprocess & run OCR engine
    raw_detections = []
    raw_blocks = []
    if img is not None:
        try:
            preprocessed = preprocess_image(img)
            raw_detections = run_ocr(preprocessed)
            raw_blocks = extract_text_blocks(raw_detections)
        except Exception as exc:
            print(f"[OCR] Execution warning: {exc}")

    # 2. Identify & parse MRZ lines
    l1, l2 = identify_mrz_lines(raw_detections)
    mrz_data = {}
    if l1 or l2:
        try:
            mrz_data = parse_mrz(l1, l2)
        except Exception as exc:
            print(f"[OCR] MRZ parse warning: {exc}")

    # 3. Extract visual zone fields
    visual_data = {}
    if raw_blocks:
        try:
            visual_data = extract_visual_zone_fields(raw_blocks)
        except Exception as exc:
            print(f"[OCR] Visual zone extraction warning: {exc}")

    # 4. Construct field values
    surname = mrz_data.get("surname", "")
    given_names = mrz_data.get("given_names", "")
    if given_names and surname:
        full_name = f"{given_names} {surname}"
    elif surname:
        full_name = surname
    elif given_names:
        full_name = given_names
    else:
        full_name = ""

    passport_num = mrz_data.get("passport_number", "")
    p_type = mrz_data.get("document_type", "P")
    country = mrz_data.get("issuing_country") or mrz_data.get("country_code", "IND")
    nat = mrz_data.get("nationality", "IND")
    dob = mrz_data.get("date_of_birth", "")
    sex = mrz_data.get("sex", "")
    exp = mrz_data.get("date_of_expiry", "")

    p_birth = visual_data.get("place_of_birth", "")
    p_issue = visual_data.get("place_of_issue", "")
    d_issue = visual_data.get("date_of_issue", "")

    # Calculate average confidence
    conf = 0.90
    if raw_blocks:
        confs = [b.get("confidence", 0.9) for b in raw_blocks if "confidence" in b]
        if confs:  
            conf = sum(confs) / len(confs)

    return {
        "name":            _f(full_name, conf),
        "passport_number": _f(passport_num, conf),
        "passport_type":  _f(p_type, conf),
        "country_code":   _f(country, conf),
        "nationality":     _f(nat, conf),
        "date_of_birth":   _f(dob, conf),
        "place_of_birth":  _f(p_birth, conf),
        "date_of_issue":   _f(d_issue, conf),
        "place_of_issue":  _f(p_issue, conf),
        "date_of_expiry":  _f(exp, conf),
        "sex":             _f(sex, conf),
        "mrz_line1":       _f(l1, conf),
        "mrz_line2":       _f(l2, conf),
    }
