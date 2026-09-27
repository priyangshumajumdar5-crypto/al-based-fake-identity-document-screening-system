"""
Unified Screening Service — Module 3, Phase 5.

Orchestrates the full screening pipeline:
    1. Module 1  — OCR extraction
    2. Module 2  — Database validation
    3. Module 3a — ELA analysis       (Phase 1)
    4. Module 3b — Metadata analysis  (Phase 1)
    5. Module 3c — EfficientNet-B0    (Phase 3, only if checkpoint exists)
    6. Advisory risk assessment        (Phase 5)

Design rules:
  - Each module is called independently. A failure in one does NOT abort others.
  - Every module's result is recorded, including the failure reason.
  - The EfficientNet-B0 checkpoint is loaded once and cached for the process
    lifetime (module-level singleton) to avoid reloading on every request.
  - Never retrains the model. Never modifies source images.
  - Does NOT write uploaded passport data to the database.
  - Missing optional components (checkpoint, dataset) are reported clearly.

NOT TESTED — code has been written but not executed.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Paths (relative to project root, resolved at runtime) ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "data" / "model_checkpoints" / "efficientnet_b0" / "best_checkpoint.pt"
_DEFAULT_ELA_DIR    = _PROJECT_ROOT / "data" / "forensics" / "ela_output"

# ── Checkpoint singleton cache ────────────────────────────────────────────────
_checkpoint_cache: Optional[Dict[str, Any]] = None
_checkpoint_path_cached: Optional[str]      = None


def _get_cached_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load and cache the model checkpoint once per process."""
    global _checkpoint_cache, _checkpoint_path_cached
    cp = str(checkpoint_path)
    if _checkpoint_cache is not None and _checkpoint_path_cached == cp:
        return _checkpoint_cache
    try:
        from app.tampering.model import load_checkpoint
        ckpt = load_checkpoint(checkpoint_path)
        _checkpoint_cache = ckpt
        _checkpoint_path_cached = cp
        logger.info("EfficientNet-B0 checkpoint loaded and cached: %s", cp)
        return ckpt
    except Exception as exc:
        logger.warning("EfficientNet-B0 checkpoint not loaded: %s", exc)
        return None


# ── Module status constants ───────────────────────────────────────────────────
STATUS_OK          = "ok"
STATUS_FAILED      = "failed"
STATUS_UNAVAILABLE = "unavailable"


def _module_status(result: Optional[Dict[str, Any]], success_key: str = "success") -> str:
    if result is None:
        return STATUS_UNAVAILABLE
    return STATUS_OK if result.get(success_key) else STATUS_FAILED


# ── Main service ──────────────────────────────────────────────────────────────

def run_screening(
    image_input:              Union[bytes, str, Path],
    current_face_image_input: Optional[Union[bytes, str, Path]] = None,
    filename:                 str = "passport.jpg",
    checkpoint_path:          Optional[Union[str, Path]] = None,
    ela_output_dir:           Optional[Union[str, Path]] = None,
    save_ela:                 bool = True,
) -> Dict[str, Any]:
    """
    Run the full screening pipeline on a passport image.

    Args:
        image_input:              Raw image bytes OR a path to an image file (Passport Image).
        current_face_image_input: Optional raw image bytes OR path to current face photo (Module 4).
        filename:                 Original filename (for logging / report naming).
        checkpoint_path:          Path to EfficientNet-B0 checkpoint (.pt).
                                  Defaults to data/model_checkpoints/efficientnet_b0/best_checkpoint.pt.
        ela_output_dir:           Directory for ELA output images.
        save_ela:                 Save ELA visualization to disk.

    Returns:
        Unified screening report dict. See structure below.

    NOTE: NOT TESTED — requires manual execution.
    """
    report: Dict[str, Any] = {
        # ── Input ──────────────────────────────────────────────────────────────
        "screening_id":   str(uuid.uuid4()),
        "timestamp":      datetime.datetime.utcnow().isoformat() + "Z",
        "filename":       filename,
        "image_info":     {},

        # ── Module outputs ─────────────────────────────────────────────────────
        "ocr":               {"status": STATUS_UNAVAILABLE, "result": None, "error": None},
        "validation":        {"status": STATUS_UNAVAILABLE, "result": None, "error": None},
        "ela":               {"status": STATUS_UNAVAILABLE, "result": None, "error": None},
        "metadata":          {"status": STATUS_UNAVAILABLE, "result": None, "error": None},
        "model_inference":   {"status": STATUS_UNAVAILABLE, "result": None, "error": None,
                              "checkpoint_available": False},
        "face_verification": {"status": STATUS_UNAVAILABLE, "result": None, "error": None},

        # ── Risk assessment ────────────────────────────────────────────────────
        "risk_assessment":   {"status": STATUS_UNAVAILABLE, "result": None},

        # ── Overall ───────────────────────────────────────────────────────────
        "module_statuses":   {},
        "warnings":          [],
        "errors":            [],
    }

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
    ela_dir         = Path(ela_output_dir)  if ela_output_dir  else _DEFAULT_ELA_DIR

    # ── Resolve image input ───────────────────────────────────────────────────
    img_bytes: Optional[bytes] = None
    tmp_path:  Optional[str]   = None

    if isinstance(image_input, (str, Path)):
        src = Path(image_input)
        if not src.exists():
            report["errors"].append(f"Image file not found: {src}")
            return report
        img_bytes = src.read_bytes()
        filename  = filename or src.name
    elif isinstance(image_input, bytes):
        img_bytes = image_input
    else:
        report["errors"].append("image_input must be bytes, str, or Path.")
        return report

    # ── Image info ────────────────────────────────────────────────────────────
    try:
        from PIL import Image as _PIL
        pil_img = _PIL.open(io.BytesIO(img_bytes))
        report["image_info"] = {
            "format": pil_img.format or "unknown",
            "mode":   pil_img.mode,
            "width":  pil_img.width,
            "height": pil_img.height,
            "size_bytes": len(img_bytes),
        }
    except Exception as exc:
        report["warnings"].append(f"Could not read image metadata: {exc}")

    # Write bytes to a temp file for modules that need a path
    try:
        suffix = Path(filename).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(img_bytes)
        tmp.close()
        tmp_path = tmp.name
    except Exception as exc:
        report["errors"].append(f"Failed to create temp file: {exc}")
        return report

    try:
        # ── Module 1: OCR ─────────────────────────────────────────────────────
        try:
            from app.database.database import initialize_database
            from app.ocr.field_extractor import extract_all_fields
            from app.utils.image_utils import load_image_from_bytes

            initialize_database()
            pil_for_ocr = load_image_from_bytes(img_bytes)
            raw_fields  = extract_all_fields(pil_for_ocr, use_template=True)

            # Normalise to {key: {value, confidence}}
            ocr_dict: Dict[str, Any] = {}
            for k, v in raw_fields.items():
                if isinstance(v, dict):
                    ocr_dict[k] = {"value": v.get("value", ""), "confidence": v.get("confidence", 0.0)}
                else:
                    ocr_dict[k] = {"value": str(v) if v else "", "confidence": 0.0}

            report["ocr"] = {"status": STATUS_OK, "result": ocr_dict, "error": None}
            logger.info("screening_id=%s OCR OK", report["screening_id"])
        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            report["ocr"] = {"status": STATUS_FAILED, "result": None, "error": str(exc)}
            report["errors"].append(f"Module 1 OCR: {exc}")

        # ── Module 2: Validation ──────────────────────────────────────────────
        ocr_ok = report["ocr"]["status"] == STATUS_OK
        if ocr_ok:
            try:
                from app.validation.validator import validate_passport
                val_result = validate_passport(ocr_dict)
                val_dict   = val_result.model_dump()
                report["validation"] = {"status": STATUS_OK, "result": val_dict, "error": None}
                logger.info("screening_id=%s Validation OK: %s", report["screening_id"],
                            val_dict.get("overall_status"))
            except Exception as exc:
                logger.error("Validation failed: %s", exc)
                report["validation"] = {"status": STATUS_FAILED, "result": None, "error": str(exc)}
                report["errors"].append(f"Module 2 Validation: {exc}")
        else:
            report["validation"]["error"] = "Skipped — OCR did not succeed."
            report["warnings"].append("Module 2 validation skipped: OCR failed.")

        # ── Module 3a: ELA ────────────────────────────────────────────────────
        try:
            from app.tampering.ela_analyzer import run_ela
            ela_save: Optional[Path] = None
            if save_ela:
                ela_dir.mkdir(parents=True, exist_ok=True)
                stem     = Path(filename).stem
                ela_save = ela_dir / f"{stem}_ela_{report['screening_id'][:8]}.png"

            ela_result = run_ela(source=tmp_path, save_path=ela_save)

            # Strip non-serialisable PIL/numpy objects
            ela_serialisable = {
                k: v for k, v in ela_result.items()
                if k not in ("ela_image", "ela_array")
            }
            report["ela"] = {
                "status": STATUS_OK if ela_result.get("success") else STATUS_FAILED,
                "result": ela_serialisable,
                "error":  ela_result.get("error"),
            }
        except Exception as exc:
            logger.error("ELA failed: %s", exc)
            report["ela"] = {"status": STATUS_FAILED, "result": None, "error": str(exc)}
            report["errors"].append(f"Module 3 ELA: {exc}")

        # ── Module 3b: Metadata ───────────────────────────────────────────────
        try:
            from app.tampering.metadata_analyzer import run_metadata_analysis
            meta_result = run_metadata_analysis(tmp_path)
            meta_serialisable = {
                k: v for k, v in meta_result.items()
                if k not in ("pil_image",)
            }
            report["metadata"] = {
                "status": STATUS_OK if meta_result.get("success") else STATUS_FAILED,
                "result": meta_serialisable,
                "error":  meta_result.get("error"),
            }
        except Exception as exc:
            logger.error("Metadata analysis failed: %s", exc)
            report["metadata"] = {"status": STATUS_FAILED, "result": None, "error": str(exc)}
            report["errors"].append(f"Module 3 Metadata: {exc}")

        # ── Module 3c: EfficientNet-B0 inference ──────────────────────────────
        ckpt_available = checkpoint_path.exists()
        report["model_inference"]["checkpoint_available"] = ckpt_available

        if ckpt_available:
            try:
                _get_cached_checkpoint(checkpoint_path)   # warm cache
                from app.tampering.inference import predict
                inf_result = predict(
                    image_path=tmp_path,
                    checkpoint_path=checkpoint_path,
                )
                # Remove non-serialisable keys (none expected, but guard)
                report["model_inference"] = {
                    "status":               STATUS_OK if inf_result.get("success") else STATUS_FAILED,
                    "result":               inf_result,
                    "error":                None if inf_result.get("success") else inf_result.get("errors"),
                    "checkpoint_available": True,
                }
            except Exception as exc:
                logger.error("EfficientNet inference failed: %s", exc)
                report["model_inference"] = {
                    "status":               STATUS_FAILED,
                    "result":               None,
                    "error":                str(exc),
                    "checkpoint_available": True,
                }
                report["errors"].append(f"Module 3 Inference: {exc}")
        else:
            report["model_inference"]["error"] = (
                f"Checkpoint not found at {checkpoint_path}. "
                "Train the EfficientNet-B0 model (Phase 3 notebook) to enable inference."
            )
            report["warnings"].append(
                "EfficientNet-B0 inference unavailable — checkpoint not found. "
                "Risk assessment will proceed without model signal."
            )

        # ── Module 4: Face Verification with Aging Context ───────────────────
        if current_face_image_input is not None:
            try:
                from app.face_verification.service import run_face_verification_pipeline

                # Extract optional passport_number and date_of_issue from OCR/Validation
                ocr_res = (report.get("ocr") or {}).get("result") or {}
                val_res = (report.get("validation") or {}).get("result") or {}

                pn = (
                    val_res.get("passport_number") or
                    (ocr_res.get("passport_number") or {}).get("value")
                )
                doi = (ocr_res.get("date_of_issue") or {}).get("value")

                fv_result = run_face_verification_pipeline(
                    current_photo_input=current_face_image_input,
                    passport_number=pn,
                    ocr_issue_date=doi,
                )

                report["face_verification"] = {
                    "status": STATUS_OK,
                    "result": fv_result,
                    "error":  None,
                }
                logger.info("screening_id=%s Face Verification OK: %s",
                            report["screening_id"], fv_result.get("face_verification_status"))
            except Exception as exc:
                logger.error("Module 4 Face Verification failed: %s", exc)
                report["face_verification"] = {
                    "status": STATUS_FAILED,
                    "result": None,
                    "error":  str(exc),
                }
                report["errors"].append(f"Module 4 Face Verification: {exc}")
        else:
            report["face_verification"] = {
                "status": STATUS_UNAVAILABLE,
                "result": None,
                "error":  "Current face photo not provided.",
            }

        # ── Risk assessment ───────────────────────────────────────────────────
        try:
            from app.services.risk_assessment import assess_risk

            val_for_risk  = report["validation"]["result"]
            ela_for_risk  = report["ela"]["result"]
            meta_for_risk = report["metadata"]["result"]
            inf_for_risk  = report["model_inference"]["result"]
            face_for_risk = report["face_verification"]["result"]

            risk = assess_risk(
                validation_result=val_for_risk,
                ela_result=ela_for_risk,
                metadata_result=meta_for_risk,
                inference_result=inf_for_risk,
                face_verification_result=face_for_risk,
            )
            report["risk_assessment"] = {
                "status": STATUS_OK,
                "result": risk.to_dict(),
            }
        except Exception as exc:
            logger.error("Risk assessment failed: %s", exc)
            report["risk_assessment"] = {
                "status": STATUS_FAILED,
                "result": None,
                "error":  str(exc),
            }

    finally:
        # ── Clean up temp file ────────────────────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Module status summary ─────────────────────────────────────────────────
    report["module_statuses"] = {
        "ocr":              report["ocr"]["status"],
        "validation":       report["validation"]["status"],
        "ela":              report["ela"]["status"],
        "metadata":         report["metadata"]["status"],
        "model_inference":  report["model_inference"]["status"],
        "face_verification":report["face_verification"]["status"],
        "risk_assessment":  report["risk_assessment"]["status"],
    }

    return report


# ── Export helpers ────────────────────────────────────────────────────────────

def export_report_json(report: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Export the screening report as JSON. Skips non-serialisable values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(i) for i in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_safe(report), f, indent=2)


def export_report_csv(report: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Export a flat summary of the screening report as CSV."""
    import csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    risk = (report.get("risk_assessment") or {}).get("result") or {}
    val  = (report.get("validation") or {}).get("result") or {}
    ela  = (report.get("ela") or {}).get("result") or {}
    inf  = (report.get("model_inference") or {}).get("result") or {}
    face = (report.get("face_verification") or {}).get("result") or {}

    row = {
        "screening_id":          report.get("screening_id"),
        "timestamp":             report.get("timestamp"),
        "filename":              report.get("filename"),
        "ocr_status":            report.get("module_statuses", {}).get("ocr"),
        "validation_status":     report.get("module_statuses", {}).get("validation"),
        "validation_overall":    val.get("overall_status"),
        "db_match":              val.get("db_match"),
        "ela_status":            report.get("module_statuses", {}).get("ela"),
        "ela_mean":              ela.get("mean_ela"),
        "ela_max":               ela.get("max_ela"),
        "metadata_status":       report.get("module_statuses", {}).get("metadata"),
        "model_status":          report.get("module_statuses", {}).get("model_inference"),
        "model_predicted_label": inf.get("predicted_label"),
        "model_score_genuine":   inf.get("score_genuine"),
        "model_score_tampered":  inf.get("score_tampered"),
        "face_verification_status": report.get("module_statuses", {}).get("face_verification"),
        "face_status":           face.get("face_verification_status"),
        "face_raw_distance":     face.get("raw_distance"),
        "face_threshold":        face.get("model_threshold"),
        "face_derived_similarity": face.get("derived_similarity_score"),
        "face_policy_case":      face.get("policy_case"),
        "risk_level":            risk.get("risk_level"),
        "risk_score":            risk.get("risk_score"),
        "db_concerns_count":      len(risk.get("db_validation_concerns", [])),
        "tampering_indicators_count": len(risk.get("tampering_indicators", [])),
        "manual_review":         risk.get("manual_review_recommended"),
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
