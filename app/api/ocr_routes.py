"""
OCR API routes — Module 1.

POST /api/v1/ocr
    Accepts a passport image upload.
    Returns structured OCR result including fields, MRZ, and confidence.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.logging_config import get_logger
from app.ocr.engine import extract_text_blocks, get_ocr_engine
from app.ocr.field_extractor import extract_all_fields
from app.ocr.mrz_parser import check_mrz_consistency, parse_mrz
from app.ocr.preprocessing import preprocess_image
from app.ocr.schemas import (
    DocumentFields,
    MRZCheckDigits,
    MRZCheckDigitResult,
    MRZResult,
    OCRFieldResult,
    OCRResponse,
)
from app.utils.image_utils import load_image_from_bytes, validate_image_format

router = APIRouter(prefix="/api/v1", tags=["OCR — Module 1"])
logger = get_logger(__name__)


def _build_field(raw_dict: dict) -> OCRFieldResult:
    """Convert raw field dict to OCRFieldResult."""
    return OCRFieldResult(
        value=raw_dict.get("value", ""),
        confidence=float(raw_dict.get("confidence", 0.0)),
        raw=raw_dict.get("raw", ""),
    )


def _build_mrz_result(mrz_data: dict, visible_fields: dict) -> MRZResult:
    """Convert raw MRZ parse dict to MRZResult schema."""
    cd_raw = mrz_data.get("check_digits", {})

    def _cd(key: str) -> MRZCheckDigitResult:
        d = cd_raw.get(key, {})
        return MRZCheckDigitResult(
            value=str(d.get("value", "")),
            expected=str(d.get("expected", "")),
            valid=bool(d.get("valid", False)),
        )

    check_digits = MRZCheckDigits(
        passport_number=_cd("passport_number"),
        date_of_birth=_cd("date_of_birth"),
        date_of_expiry=_cd("date_of_expiry"),
        optional_data=_cd("optional_data"),
        composite=_cd("composite"),
    )

    # Run MRZ consistency check
    consistency = check_mrz_consistency(mrz_data, visible_fields)

    return MRZResult(
        line1=mrz_data.get("line1", ""),
        line2=mrz_data.get("line2", ""),
        document_type=mrz_data.get("document_type", ""),
        issuing_country=mrz_data.get("issuing_country", ""),
        surname=mrz_data.get("surname", ""),
        given_names=mrz_data.get("given_names", ""),
        passport_number=mrz_data.get("passport_number", ""),
        nationality=mrz_data.get("nationality", ""),
        date_of_birth=mrz_data.get("date_of_birth", ""),
        sex=mrz_data.get("sex", ""),
        date_of_expiry=mrz_data.get("date_of_expiry", ""),
        optional_data=mrz_data.get("optional_data", ""),
        check_digits=check_digits,
        all_check_digits_valid=bool(mrz_data.get("all_check_digits_valid", False)),
        parse_errors=mrz_data.get("parse_errors", []),
        consistency=consistency,
    )


@router.post(
    "/ocr",
    response_model=OCRResponse,
    summary="Extract passport fields via OCR",
    description=(
        "Upload a passport image (JPEG/PNG/BMP) to extract structured fields "
        "including name, passport number, dates, MRZ, and OCR confidence scores. "
        "**PROTOTYPE** — uses synthetic data only."
    ),
)
async def ocr_passport(
    file: UploadFile = File(..., description="Passport image file."),
    use_template: bool = Form(
        default=True,
        description="Use template-based field extraction (recommended for synthetic passports).",
    ),
) -> OCRResponse:
    """OCR endpoint — Module 1."""
    logger.info("OCR request received: filename=%s", file.filename)

    # ── Validate upload ────────────────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}. Upload an image.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Load image ─────────────────────────────────────────────────────────────
    try:
        img = load_image_from_bytes(raw_bytes)
    except ValueError as exc:
        logger.warning("Image load failed: %s", exc)
        return OCRResponse(
            success=False,
            error=f"Cannot decode image: {exc}",
        )

    valid, msg = validate_image_format(img)
    if not valid:
        return OCRResponse(success=False, error=msg)

    # ── Run OCR ────────────────────────────────────────────────────────────────
    logger.info("Running OCR on image (%dx%d).", *img.size)
    warnings = []

    try:
        raw_fields = extract_all_fields(img, use_template=use_template)
    except Exception as exc:
        logger.error("OCR field extraction failed: %s", exc)
        return OCRResponse(
            success=False,
            error=f"OCR processing failed: {exc}",
        )

    # ── Build structured fields ────────────────────────────────────────────────
    def get_field(key: str) -> OCRFieldResult:
        d = raw_fields.get(key, {})
        if isinstance(d, dict):
            return _build_field(d)
        return OCRFieldResult(value=str(d) if d else "", confidence=0.0)

    doc_fields = DocumentFields(
        name            =get_field("name"),
        passport_number =get_field("passport_number"),
        passport_type   =get_field("passport_type"),
        country_code    =get_field("country_code"),
        nationality     =get_field("nationality"),
        date_of_birth   =get_field("date_of_birth"),
        place_of_birth  =get_field("place_of_birth"),
        date_of_issue   =get_field("date_of_issue"),
        place_of_issue  =get_field("place_of_issue"),
        date_of_expiry  =get_field("date_of_expiry"),
        sex             =get_field("sex"),
    )

    # ── MRZ parsing ────────────────────────────────────────────────────────────
    mrz_result: Optional[MRZResult] = None
    mrz_line1 = raw_fields.get("mrz_line1", {})
    mrz_line2 = raw_fields.get("mrz_line2", {})
    l1 = mrz_line1.get("value", "") if isinstance(mrz_line1, dict) else str(mrz_line1)
    l2 = mrz_line2.get("value", "") if isinstance(mrz_line2, dict) else str(mrz_line2)

    if l1 or l2:
        try:
            mrz_data = parse_mrz(l1, l2)
            mrz_result = _build_mrz_result(mrz_data, raw_fields)
            if not mrz_result.all_check_digits_valid:
                warnings.append("MRZ check digit validation failed — manual review recommended.")
        except Exception as exc:
            logger.warning("MRZ parsing failed: %s", exc)
            warnings.append(f"MRZ parsing failed: {exc}")
    else:
        warnings.append("MRZ lines not detected in this image.")

    # ── Raw text (full page) for debugging ────────────────────────────────────
    preprocessed = preprocess_image(img)
    engine = get_ocr_engine()
    raw_detections = engine.read(preprocessed)
    raw_text_blocks = extract_text_blocks(raw_detections)

    logger.info("OCR complete — %d raw text blocks detected.", len(raw_text_blocks))

    return OCRResponse(
        success=True,
        document_type="passport",
        fields=doc_fields,
        mrz=mrz_result,
        raw_text=raw_text_blocks,
        warnings=warnings,
    )
