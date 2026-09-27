"""
Validation API routes — Module 2.

POST /api/v1/validate
    Accepts structured OCR fields and returns validation result.

POST /api/v1/screen
    Accepts a passport image, runs OCR + Validation, returns combined result.

Future integration points for Module 3 and Module 4 are marked but NOT implemented.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.logging_config import get_logger
from app.ocr.field_extractor import extract_all_fields
from app.ocr.mrz_parser import check_mrz_consistency, parse_mrz
from app.ocr.schemas import OCRResponse
from app.utils.image_utils import load_image_from_bytes, validate_image_format
from app.validation.schemas import ValidationResponse
from app.validation.validator import validate_document

router = APIRouter(prefix="/api/v1", tags=["Validation — Module 2"])
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request model for /validate
# ─────────────────────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    """
    Input for the /validate endpoint.
    Accepts the fields and MRZ from an OCR result.
    """
    fields: Dict[str, Any]
    mrz: Optional[Dict[str, Any]] = None


class ScreeningResult(BaseModel):
    """
    Combined result from the /screen endpoint.
    Returned after running OCR → Validation pipeline.
    """
    ocr: OCRResponse
    validation: ValidationResponse

    # ── Future integration points ─────────────────────────────────────────────
    # Module 3 — Tampering Detection (NOT IMPLEMENTED)
    tampering_detection: Optional[str] = "NOT_IMPLEMENTED"

    # Module 4 — Face Verification (NOT IMPLEMENTED)
    face_verification: Optional[str] = "NOT_IMPLEMENTED"


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/validate
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate extracted passport document data",
    description=(
        "Accepts structured document fields (from OCR output) and runs "
        "all Module 2 validation rules: required fields, dates, passport number, "
        "expiry, cross-field consistency, MRZ check digits, and database lookup. "
        "**PROTOTYPE** — uses synthetic database only."
    ),
)
async def validate_passport(body: ValidateRequest) -> ValidationResponse:
    """Validation endpoint — Module 2."""
    logger.info(
        "Validation request received — passport_number=%s",
        body.fields.get("passport_number", {}) if isinstance(body.fields.get("passport_number"), dict)
        else body.fields.get("passport_number", "unknown"),
    )

    try:
        result = validate_document(fields=body.fields, mrz=body.mrz)
    except Exception as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation processing error: {exc}",
        )

    logger.info("Validation complete — status: %s", result.validation_status)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/screen
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/screen",
    response_model=ScreeningResult,
    summary="Full screening pipeline: OCR → Validation",
    description=(
        "Upload a passport image. Runs Module 1 (OCR) followed by "
        "Module 2 (Validation) and returns a combined screening result. "
        "Modules 3 (Tampering Detection) and 4 (Face Verification) are "
        "NOT YET IMPLEMENTED. **PROTOTYPE** — synthetic data only."
    ),
)
async def screen_passport(
    file: UploadFile = File(..., description="Passport image file."),
    use_template: bool = Form(
        default=True,
        description="Use template-based OCR field extraction.",
    ),
) -> ScreeningResult:
    """
    Orchestrates Module 1 → Module 2.

    Future integration points:
        Module 3: run_tampering_detection(img) — NOT IMPLEMENTED
        Module 4: run_face_verification(img, portrait) — NOT IMPLEMENTED
    """
    logger.info("Screen request received: filename=%s", file.filename)

    # ── Load image ─────────────────────────────────────────────────────────────
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        img = load_image_from_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    valid, msg = validate_image_format(img)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    # ── Module 1: OCR ─────────────────────────────────────────────────────────
    logger.info("Running Module 1 — OCR.")
    warnings = []

    try:
        raw_fields = extract_all_fields(img, use_template=use_template)
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    # Build OCR response (reuse logic from ocr_routes via direct function calls)
    from app.api.ocr_routes import _build_field, _build_mrz_result
    from app.ocr.schemas import DocumentFields, OCRFieldResult
    from app.ocr.engine import extract_text_blocks, get_ocr_engine
    from app.ocr.preprocessing import preprocess_image

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

    mrz_result = None
    mrz_line1 = raw_fields.get("mrz_line1", {})
    mrz_line2 = raw_fields.get("mrz_line2", {})
    l1 = mrz_line1.get("value", "") if isinstance(mrz_line1, dict) else str(mrz_line1)
    l2 = mrz_line2.get("value", "") if isinstance(mrz_line2, dict) else str(mrz_line2)

    mrz_dict: Optional[Dict] = None
    if l1 or l2:
        try:
            mrz_dict = parse_mrz(l1, l2)
            mrz_dict["consistency"] = check_mrz_consistency(mrz_dict, raw_fields)
            mrz_result = _build_mrz_result(mrz_dict, raw_fields)
        except Exception as exc:
            warnings.append(f"MRZ parsing failed: {exc}")
    else:
        warnings.append("MRZ not detected.")

    preprocessed = preprocess_image(img)
    engine = get_ocr_engine()
    raw_detections = engine.read(preprocessed)
    raw_text_blocks = extract_text_blocks(raw_detections)

    ocr_response = OCRResponse(
        success=True,
        document_type="passport",
        fields=doc_fields,
        mrz=mrz_result,
        raw_text=raw_text_blocks,
        warnings=warnings,
    )

    # ── Module 2: Validation ───────────────────────────────────────────────────
    logger.info("Running Module 2 — Validation.")
    # Convert fields to dicts for validator
    fields_dict = {
        k: {"value": getattr(doc_fields, k).value, "confidence": getattr(doc_fields, k).confidence}
        for k in doc_fields.model_fields
        if hasattr(doc_fields, k) and isinstance(getattr(doc_fields, k), OCRFieldResult)
    }

    mrz_dict_for_validation = mrz_dict if mrz_dict else None

    validation_result = validate_document(fields=fields_dict, mrz=mrz_dict_for_validation)

    # ── Return combined result ─────────────────────────────────────────────────
    # Future: Module 3 (tampering) and Module 4 (face verification) plug in here.
    return ScreeningResult(
        ocr=ocr_response,
        validation=validation_result,
        tampering_detection="NOT_IMPLEMENTED",
        face_verification="NOT_IMPLEMENTED",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/validate-passport  (Module 2 — real DB comparison)
# ─────────────────────────────────────────────────────────────────────────────

class PassportValidationResponse(BaseModel):
    """Combined OCR + per-field DB validation result."""
    ocr_result: Dict[str, Any]
    validation_result: Dict[str, Any]


@router.post(
    "/validate-passport",
    response_model=PassportValidationResponse,
    summary="Upload passport image → OCR → Database field comparison",
    description=(
        "Upload a passport image. Runs Module 1 (OCR) to extract fields, "
        "then looks up the passport number in the real CSV-seeded database "
        "and compares each field (MATCHED / MISMATCH / NOT_VERIFIED). "
        "Returns OCR results and per-field validation for the officer dashboard."
    ),
)
async def validate_passport_endpoint(
    file: UploadFile = File(..., description="Passport image file (JPEG/PNG)."),
) -> PassportValidationResponse:
    """OCR → DB lookup → per-field comparison pipeline."""
    logger.info("validate-passport request: filename=%s", file.filename)

    # ── Read and validate image ───────────────────────────────────────────────
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        img = load_image_from_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image: {exc}",
        )

    valid, msg = validate_image_format(img)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    # ── Module 1: OCR extraction ──────────────────────────────────────────────
    try:
        raw_fields = extract_all_fields(img, use_template=True)
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {exc}",
        )

    # Build a serializable OCR dict
    ocr_dict = {}
    for key, val in raw_fields.items():
        if isinstance(val, dict):
            ocr_dict[key] = {
                "value": val.get("value", ""),
                "confidence": val.get("confidence", 0.0),
            }
        else:
            ocr_dict[key] = {"value": str(val) if val else "", "confidence": 0.0}

    # ── Module 2: Database validation ─────────────────────────────────────────
    from app.validation.validator import validate_passport as _validate_passport

    try:
        val_result = _validate_passport(raw_fields)
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        )

    logger.info(
        "validate-passport complete: status=%s, passport=%s",
        val_result.overall_status,
        val_result.passport_number,
    )

    return PassportValidationResponse(
        ocr_result=ocr_dict,
        validation_result=val_result.model_dump(),
    )

