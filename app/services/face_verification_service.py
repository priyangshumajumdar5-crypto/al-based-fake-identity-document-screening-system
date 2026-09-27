"""
Module 4 — Face Verification Service Wrapper.

Exposes top-level functions for Module 4 integration into notebooks,
FastAPI endpoints, and full pipeline screening services.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.face_verification.service import run_face_verification_pipeline


def verify_faces_with_context(
    passport_photo_path: Union[str, Path, bytes],
    current_face_photo_path: Union[str, Path, bytes],
    passport_number: Optional[str] = None,
    ocr_issue_date: Optional[str] = None,
    reference_date: Optional[date] = None,
    model_name: Optional[str] = None,
    detector_backend: Optional[str] = None,
    custom_threshold: Optional[float] = None,
    aging_threshold_years: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Module 4 Face Verification with Aging-Aware Context.

    Args:
        passport_photo_path: Path or bytes for passport photo (Image A).
        current_face_photo_path: Path or bytes for current face photo (Image B).
        passport_number: Extracted passport number for database lookup (optional).
        ocr_issue_date: Extracted issue date string from OCR (optional).
        reference_date: Reference date for elapsed time calculation (optional).
        model_name: Face model (default 'ArcFace').
        detector_backend: Detector backend (default 'retinaface').
        custom_threshold: Custom distance threshold (optional).
        aging_threshold_years: Aging threshold in years (default 5.0).

    Returns:
        Structured result dict containing face verification & age context findings.
    """
    return run_face_verification_pipeline(
        passport_photo_input=passport_photo_path,
        current_photo_input=current_face_photo_path,
        passport_number=passport_number,
        ocr_issue_date=ocr_issue_date,
        reference_date=reference_date,
        model_name=model_name,
        detector_backend=detector_backend,
        custom_threshold=custom_threshold,
        aging_threshold_years=aging_threshold_years,
    )
