"""
Module 4 — Unified Face Verification Pipeline Service.

Orchestrates all Module 4 responsibilities:
  1. Database record lookup (SQLite)
  2. Issue date resolution & elapsed time calculation
  3. Image validation & face count checking
  4. Pairwise face verification (DeepFace ArcFace)
  5. Age-related context policy evaluation
  6. Structured report generation

Database lookup failures do NOT short-circuit face verification.
"""

from __future__ import annotations

import datetime
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.core.config import settings
from app.core.logging_config import get_logger
from app.face_verification.age_policy import AgePolicyConfig, apply_age_context_policy
from app.face_verification.date_resolver import resolve_issue_date
from app.face_verification.db_lookup import lookup_passport_record
from app.face_verification.verifier import verify_faces

logger = get_logger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_face_verification_pipeline(
    current_photo_input: Union[str, Path, bytes],
    passport_number: Optional[str] = None,
    passport_photo_input: Optional[Union[str, Path, bytes]] = None,
    ocr_issue_date: Optional[str] = None,
    reference_date: Optional[date] = None,
    model_name: Optional[str] = None,
    detector_backend: Optional[str] = None,
    distance_metric: Optional[str] = None,
    custom_threshold: Optional[float] = None,
    aging_threshold_years: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run Module 4 Database-Based Face Verification pipeline.

    Workflow:
      1. Lookup passport record in SQLite by passport_number.
      2. Retrieve reference portrait associated with that database record.
      3. Compare Database Portrait vs Current/Live Face Photo using DeepFace (ArcFace).
      4. Evaluate issue date & elapsed time policy.

    Args:
        current_photo_input: Current/Live face photo uploaded by officer (Image B).
        passport_number: Extracted passport number for SQLite lookup.
        passport_photo_input: Optional override portrait path/bytes (used only if provided & DB lookup unsupplied).
        ocr_issue_date: Optional extracted issue date string from OCR.
        reference_date: Optional reference date for elapsed time calculation.
        model_name: DeepFace model name (defaults to 'ArcFace').
        detector_backend: Detector backend (defaults to 'retinaface').
        distance_metric: Distance metric (defaults to 'cosine').
        custom_threshold: Custom verification distance threshold override.
        aging_threshold_years: Aging context threshold in years (defaults to 5.0).

    Returns:
        Structured pipeline report dict containing database portrait findings.
    """
    ref_date = reference_date or date.today()

    m_name = model_name or getattr(settings, "FACE_MODEL_NAME", "ArcFace")
    d_backend = detector_backend or getattr(settings, "FACE_DETECTOR_BACKEND", "retinaface")
    d_metric = distance_metric or getattr(settings, "FACE_DISTANCE_METRIC", "cosine")
    aging_thresh = aging_threshold_years or getattr(settings, "FACE_AGING_THRESHOLD_YEARS", 5.0)

    report: Dict[str, Any] = {
        "pipeline_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_date": ref_date.isoformat(),

        # ── Passport & DB Lookup ──────────────────────────────────────────────
        "passport_number_provided": passport_number,
        "passport_number_extraction_status": "EXTRACTED" if passport_number else "MISSING",
        "database_lookup_status": "NOT_RUN",
        "database_record_found": False,
        "database_table_source": None,
        "database_portrait_path": None,
        "database_error": None,

        # ── Date Resolution & Age ──────────────────────────────────────────────
        "resolved_issue_date": "UNAVAILABLE",
        "issue_date_source": "UNAVAILABLE",
        "elapsed_years": None,
        "elapsed_days": None,
        "date_discrepancy_warning": None,

        # ── Face Verification ──────────────────────────────────────────────────
        "face_verification_status": "UNAVAILABLE",
        "is_match": None,
        "raw_distance": None,
        "model_threshold": None,
        "derived_similarity_score": None,
        "passport_face_count": 0,
        "current_face_count": 0,
        "model_name": m_name,
        "detector_backend": d_backend,
        "distance_metric": d_metric,

        # ── Age Context Policy ─────────────────────────────────────────────────
        "policy_case": "VERIFICATION_UNAVAILABLE",
        "age_context_status": "VERIFICATION_UNAVAILABLE",
        "manual_review_recommended": True,
        "human_readable_explanation": "",
        "warnings": [],
        "errors": [],
    }

    # 1. Step 1 & 2: Database Record Lookup
    db_status, db_record, table_source, db_err = lookup_passport_record(passport_number)
    report["database_lookup_status"] = db_status
    report["database_record_found"] = (db_status == "FOUND")
    report["database_table_source"] = table_source
    report["database_error"] = db_err
    if db_err:
        report["warnings"].append(f"Database lookup error: {db_err}")

    # 2. Step 3: Issue Date Resolution
    parsed_date, date_str, date_src, elapsed_y, elapsed_d, date_warn = resolve_issue_date(
        db_record=db_record,
        ocr_issue_date=ocr_issue_date,
        reference_date=ref_date,
    )

    report["resolved_issue_date"] = date_str
    report["issue_date_source"] = date_src
    report["elapsed_years"] = elapsed_y
    report["elapsed_days"] = elapsed_d
    report["date_discrepancy_warning"] = date_warn
    if date_warn:
        report["warnings"].append(date_warn)

    # 3. Resolve Reference Database Passport Portrait
    db_portrait_target: Optional[Union[str, Path, bytes]] = None

    if db_record and isinstance(db_record, dict):
        raw_portrait_path = db_record.get("passport_photo_path")
        if raw_portrait_path:
            p_path = Path(raw_portrait_path)
            if not p_path.is_absolute():
                p_path = _PROJECT_ROOT / raw_portrait_path
            if p_path.exists() and p_path.is_file() and p_path.stat().st_size > 0:
                db_portrait_target = str(p_path)
                report["database_portrait_path"] = str(p_path)

    # Fallback to passport_photo_input only if explicit override passed and DB portrait empty
    if db_portrait_target is None and passport_photo_input is not None:
        db_portrait_target = passport_photo_input

    # 4. Perform Face Verification against Database Portrait
    if db_status != "FOUND":
        verif_res = {
            "verification_status": "DATABASE_RECORD_NOT_FOUND",
            "error_message": f"No SQLite database record found for passport number '{passport_number or 'MISSING'}'. Face verification skipped.",
        }
    elif db_portrait_target is None:
        verif_res = {
            "verification_status": "DATABASE_PORTRAIT_UNAVAILABLE",
            "error_message": f"Database record found for passport '{passport_number}', but reference portrait image is unavailable.",
        }
    else:
        verif_res = verify_faces(
            passport_photo_input=db_portrait_target,
            current_photo_input=current_photo_input,
            model_name=m_name,
            detector_backend=d_backend,
            distance_metric=d_metric,
            custom_threshold=custom_threshold,
        )

    report["face_verification_status"] = verif_res.get("verification_status", "UNAVAILABLE")
    report["is_match"] = verif_res.get("is_match")
    report["raw_distance"] = verif_res.get("raw_distance")
    report["model_threshold"] = verif_res.get("threshold")
    report["derived_similarity_score"] = verif_res.get("derived_similarity_score")
    report["passport_face_count"] = verif_res.get("passport_face_count", 0)
    report["current_face_count"] = verif_res.get("current_face_count", 0)
    report["primary_portrait_extracted"] = verif_res.get("primary_portrait_extracted", False)
    report["passport_candidate_info"] = verif_res.get("passport_candidate_info", [])
    report["current_candidate_info"] = verif_res.get("current_candidate_info", [])

    if verif_res.get("error_message"):
        report["errors"].append(verif_res["error_message"])

    # 5. Apply Age-Related Context Policy
    policy_cfg = AgePolicyConfig(
        aging_threshold_years=aging_thresh,
    )

    policy_res = apply_age_context_policy(
        verification_result=verif_res,
        resolved_issue_date=parsed_date,
        elapsed_years=elapsed_y,
        issue_date_source=date_src,
        config=policy_cfg,
    )

    report["policy_case"] = policy_res.get("policy_case")
    report["age_context_status"] = policy_res.get("age_context_status")
    report["manual_review_recommended"] = policy_res.get("manual_review_recommended", True)
    report["human_readable_explanation"] = policy_res.get("explanation", "")

    logger.info(
        "Module 4 Database-Based Pipeline complete: pipeline_id=%s, db_status=%s, verif_status=%s, policy_case=%s",
        report["pipeline_id"], db_status, report["face_verification_status"], report["policy_case"],
    )

    return report
