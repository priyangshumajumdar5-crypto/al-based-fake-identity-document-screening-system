"""
Tampering Pipeline — Module 3, Phase 1.

Orchestrates ELA + Metadata analysis into a single structured forensic report.
Does NOT produce a fraud decision, risk score, or genuine/tampered label.

Phase 2+ (EfficientNet-B0, deepfake detection) will extend this pipeline.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.tampering.ela_analyzer import run_ela
from app.tampering.metadata_analyzer import run_metadata_analysis

# Default directory for ELA output images
_DEFAULT_ELA_OUTPUT_DIR = "data/forensics/ela_output"


def run_forensic_pipeline(
    image_path: Union[str, Path],
    ela_quality: int = 90,
    ela_scale: int = 15,
    ela_output_dir: Optional[Union[str, Path]] = None,
    save_ela: bool = True,
) -> Dict[str, Any]:
    """
    Run ELA + Metadata forensic analysis on a passport image.

    Args:
        image_path:     Path to the passport image file.
        ela_quality:    JPEG recompression quality for ELA (1-95). Default 90.
        ela_scale:      Amplification factor for ELA visualisation. Default 15.
        ela_output_dir: Directory to save ELA images. Defaults to
                        'data/forensics/ela_output/'.
        save_ela:       Whether to save the ELA image to disk. Default True.

    Returns:
        Structured report dict with keys:
            status          ("ok" | "partial" | "failed")
            timestamp       (ISO 8601 string)
            input           (dict)   — resolved path + filename
            metadata        (dict)   — full metadata result
            ela             (dict)   — ELA result (without the PIL image object)
            warnings        (list)   — non-fatal issues
            errors          (list)   — fatal issues per sub-module

    NOTE: This phase produces NO fraud score, risk level, or tampered/genuine label.
    """
    report: Dict[str, Any] = {
        "status":    "failed",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input":     {},
        "metadata":  {},
        "ela":       {},
        "warnings":  [],
        "errors":    [],
    }

    # ── Validate input path ───────────────────────────────────────────────────
    image_path = Path(image_path)
    if not image_path.exists():
        report["errors"].append(f"Image not found: {image_path}")
        return report

    report["input"] = {
        "path":     str(image_path.resolve()),
        "filename": image_path.name,
    }

    # ── Determine ELA output path ─────────────────────────────────────────────
    ela_save_path: Optional[Path] = None
    if save_ela:
        out_dir = Path(ela_output_dir) if ela_output_dir else Path(_DEFAULT_ELA_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        ela_save_path = out_dir / f"{stem}_ela_q{ela_quality}.png"

    # ── Run Metadata Analysis ─────────────────────────────────────────────────
    meta_result = run_metadata_analysis(image_path)
    report["metadata"] = {
        "success":      meta_result["success"],
        "image_info":   meta_result.get("image_info", {}),
        "exif_fields":  meta_result.get("exif_fields", {}),
        "observations": meta_result.get("observations", []),
        "error":        meta_result.get("error"),
    }
    if meta_result.get("error"):
        report["errors"].append(f"Metadata analysis: {meta_result['error']}")

    # ── Run ELA ───────────────────────────────────────────────────────────────
    ela_result = run_ela(
        source=image_path,
        quality=ela_quality,
        scale=ela_scale,
        save_path=ela_save_path,
    )

    # Strip the PIL Image / ndarray objects — not JSON-serialisable
    ela_summary: Dict[str, Any] = {
        "success":       ela_result["success"],
        "mean_ela":      ela_result.get("mean_ela"),
        "max_ela":       ela_result.get("max_ela"),
        "quality_used":  ela_result.get("quality_used"),
        "scale_used":    ela_result.get("scale_used"),
        "source_format": ela_result.get("source_format"),
        "converted":     ela_result.get("converted", False),
        "ela_image_path": ela_result.get("save_path"),
        "error":          ela_result.get("error"),
    }
    report["ela"] = ela_summary

    if ela_result.get("converted"):
        report["warnings"].append(
            f"Source image is {ela_result['source_format']}. "
            "It was converted to JPEG for ELA. Results reflect the converted copy, "
            "not the original file format."
        )
    if ela_result.get("error"):
        report["errors"].append(f"ELA analysis: {ela_result['error']}")

    # ── Determine overall status ──────────────────────────────────────────────
    if meta_result["success"] and ela_result["success"]:
        report["status"] = "ok"
    elif meta_result["success"] or ela_result["success"]:
        report["status"] = "partial"
    else:
        report["status"] = "failed"

    return report
