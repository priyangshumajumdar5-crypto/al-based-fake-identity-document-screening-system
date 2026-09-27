"""
Module 1: Passport OCR Extraction Subsystem.

Exposes a single entry-point function:
    extract_passport_data(image_path: str, verbose: bool = False) -> dict
"""
from __future__ import annotations

from typing import Any, Dict, Union

from app.ocr.engine import extract_text_blocks, run_ocr
from app.ocr.field_extractor import extract_visual_zone_fields, merge_mrz_and_visual_fields
from app.ocr.mrz_parser import identify_mrz_lines, parse_mrz
from app.ocr.preprocessing import preprocess_image
from app.ocr.schemas import PassportOCRResult


def extract_passport_data(image_path: Union[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run full OCR extraction pipeline on a passport bio-data page image:
    preprocess -> EasyOCR -> MRZ identify/parse/checksum -> visual field extract -> merge.

    Args:
        image_path: File path (str/Path), PIL Image, or bytes.
        verbose: If True, prints debug match information.

    Returns:
        Structured Python dictionary matching PassportOCRResult schema.
    """
    # 1. Preprocess image (load, resize, grayscale, enhance contrast)
    preprocessed_img = preprocess_image(image_path)

    # 2. Run EasyOCR engine
    ocr_detections = run_ocr(preprocessed_img)
    ocr_blocks = extract_text_blocks(ocr_detections)
    raw_lines = [b["text"] for b in ocr_blocks if b.get("text")]

    # 3. Identify, clean, parse MRZ and validate checksums
    mrz_l1, mrz_l2 = identify_mrz_lines(ocr_detections)
    mrz_data = parse_mrz(mrz_l1, mrz_l2)

    # 4. Extract visual zone non-MRZ fields via independent label matching
    visual_data = extract_visual_zone_fields(ocr_blocks, verbose=verbose)

    # 5. Merge MRZ + visual fields into final structured dictionary
    result_dict = merge_mrz_and_visual_fields(mrz_data, visual_data, raw_lines)

    # Validate against dataclass schema
    ocr_result = PassportOCRResult(**result_dict)
    return ocr_result.to_dict()


__all__ = [
    "extract_passport_data",
    "PassportOCRResult",
]
