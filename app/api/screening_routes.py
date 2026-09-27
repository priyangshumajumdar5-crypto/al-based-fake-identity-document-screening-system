"""
Screening API — Module 3, Phase 5.

POST /api/v1/screening/analyze

Accepts a passport image upload, runs the full unified screening pipeline
(OCR + DB validation + ELA + Metadata + EfficientNet-B0 if available),
and returns the structured screening report as JSON.

All existing /api/v1/validate and /api/v1/screen routes are UNCHANGED.

NOT TESTED — requires manual execution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.logging_config import get_logger

router = APIRouter(prefix="/api/v1/screening", tags=["Screening — Phase 5"])
logger = get_logger(__name__)

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@router.post(
    "/analyze",
    summary="Full passport screening (OCR + Validation + Tampering + Face Verification + Risk)",
    response_class=JSONResponse,
)
async def analyze_passport(
    file: UploadFile = File(..., description="Passport image (JPEG / PNG / BMP / TIFF)"),
    current_face_file: Optional[UploadFile] = File(
        None,
        description="Optional current face photo for Module 4 face verification",
    ),
    checkpoint_path: Optional[str] = Query(
        None,
        description="Optional override path to EfficientNet-B0 checkpoint (.pt). "
                    "Defaults to data/model_checkpoints/efficientnet_b0/best_checkpoint.pt.",
    ),
):
    """
    Run the full Phase 5 unified screening pipeline.

    Returns a unified screening report including:
    - Module 1 OCR fields
    - Module 2 database validation result
    - Module 3 ELA analysis
    - Module 3 metadata analysis
    - Module 3 EfficientNet-B0 inference (if checkpoint available)
    - Module 4 Face Verification (if current face photo provided)
    - Advisory risk assessment

    **PROTOTYPE — NOT an official government system.**
    """
    filename = file.filename or "passport.jpg"
    suffix = Path(filename).suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    try:
        img_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not img_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    current_face_bytes: Optional[bytes] = None
    if current_face_file is not None:
        try:
            current_face_bytes = await current_face_file.read()
            if not current_face_bytes:
                current_face_bytes = None
        except Exception as exc:
            logger.warning("Could not read uploaded current_face_file: %s", exc)

    logger.info("POST /screening/analyze — file=%s size=%d bytes, face_file_provided=%s",
                filename, len(img_bytes), current_face_bytes is not None)

    try:
        from app.services.screening_service import run_screening

        report = run_screening(
            image_input=img_bytes,
            current_face_image_input=current_face_bytes,
            filename=filename,
            checkpoint_path=checkpoint_path if checkpoint_path else None,
        )
    except Exception as exc:
        logger.error("Screening pipeline exception: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening pipeline error: {exc}",
        )

    # Sanitise: do not expose absolute local paths unnecessarily
    _redact_paths(report)

    return JSONResponse(content=report)


def _redact_paths(report: dict) -> None:
    """
    Replace absolute local filesystem paths with basenames in the report.
    Prevents local path leakage in API responses.
    """
    ela = (report.get("ela") or {}).get("result") or {}
    ela_path = ela.get("save_path") or ela.get("ela_image_path")
    if ela_path and Path(ela_path).is_absolute():
        ela["save_path"] = Path(ela_path).name
        ela["ela_image_path"] = Path(ela_path).name
