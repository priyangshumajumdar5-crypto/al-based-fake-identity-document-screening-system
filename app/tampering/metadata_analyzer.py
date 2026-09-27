"""
Metadata Analyzer — Module 3, Phase 1.

Extracts image metadata (format, dimensions, EXIF) and adds cautious
rule-based observations. Observations are clearly separated from facts.
Missing metadata does NOT imply tampering.

Dependencies: Pillow (already in requirements.txt)
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ExifTags, UnidentifiedImageError

# ── EXIF tag name lookup ──────────────────────────────────────────────────────
_TAG_NAMES: Dict[int, str] = {v: k for k, v in ExifTags.TAGS.items()}

# EXIF tag IDs we want to surface explicitly
_EXIF_FIELDS_OF_INTEREST = {
    "Make":             "camera_make",
    "Model":            "camera_model",
    "Software":         "software",
    "DateTime":         "datetime",
    "DateTimeOriginal": "datetime_original",
    "DateTimeDigitized":"datetime_digitized",
    "GPSInfo":          "gps_present",
    "Orientation":      "orientation",
    "XResolution":      "x_resolution",
    "YResolution":      "y_resolution",
    "ResolutionUnit":   "resolution_unit",
    "Flash":            "flash",
    "FocalLength":      "focal_length",
    "ExifImageWidth":   "exif_width",
    "ExifImageHeight":  "exif_height",
    "UserComment":      "user_comment",
    "ImageDescription": "image_description",
    "Copyright":        "copyright",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_exif_value(val: Any) -> Any:
    """Convert EXIF value to a JSON-serialisable Python type."""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip()
        except Exception:
            return val.hex()
    if isinstance(val, tuple):
        if len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], int):
            # Rational number
            return round(val[0] / val[1], 6) if val[1] != 0 else None
        return [_safe_exif_value(v) for v in val]
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    return str(val)


def _extract_exif(img: Image.Image) -> Dict[str, Any]:
    """Return a dict of EXIF data, or an empty dict if unavailable."""
    try:
        raw_exif = img._getexif()          # type: ignore[attr-defined]
    except (AttributeError, Exception):
        raw_exif = None

    if not raw_exif:
        return {}

    exif: Dict[str, Any] = {}
    for tag_id, value in raw_exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, f"Unknown_{tag_id}")
        if tag_name == "GPSInfo":
            exif["gps_present"] = True
            continue
        try:
            exif[tag_name] = _safe_exif_value(value)
        except Exception:
            exif[tag_name] = "<unreadable>"
    return exif


def _build_structured_exif(raw_exif: Dict[str, Any]) -> Dict[str, Any]:
    """Pick relevant EXIF fields into a clean dict."""
    structured: Dict[str, Any] = {}
    for exif_key, friendly_key in _EXIF_FIELDS_OF_INTEREST.items():
        if exif_key in raw_exif:
            structured[friendly_key] = raw_exif[exif_key]
    return structured


def _generate_observations(
    fmt: str,
    structured_exif: Dict[str, Any],
    raw_exif: Dict[str, Any],
    file_size: Optional[int],
    width: int,
    height: int,
) -> List[str]:
    """
    Return a list of cautious, rule-based observations.
    Each observation is a plain-language note, NOT a fraud conclusion.
    """
    obs: List[str] = []

    if not raw_exif:
        obs.append(
            "No EXIF metadata found. "
            "This is normal for PNG files, screenshots, or images that have been "
            "re-exported through tools that strip metadata."
        )

    software = structured_exif.get("software", "")
    if software:
        editing_keywords = [
            "photoshop", "gimp", "lightroom", "affinity", "paint.net",
            "illustrator", "inkscape", "canva", "pixlr",
        ]
        if any(kw in str(software).lower() for kw in editing_keywords):
            obs.append(
                f"Software tag '{software}' indicates a photo-editing application "
                "was used at some point. This is common for legitimate document scans "
                "but warrants manual review."
            )

    if structured_exif.get("gps_present"):
        obs.append(
            "GPS metadata is embedded in the image. "
            "Passport photos typically do not contain GPS data."
        )

    cam_make = structured_exif.get("camera_make", "")
    cam_model = structured_exif.get("camera_model", "")
    if not cam_make and not cam_model and raw_exif:
        obs.append(
            "Camera make/model tags are absent despite other EXIF data being present. "
            "This can occur after metadata stripping or re-saving."
        )

    if fmt == "PNG":
        obs.append(
            "Image is in PNG format. Passport scan images are typically JPEG; "
            "PNG conversion can alter compression characteristics."
        )

    if file_size is not None and file_size < 10_000:
        obs.append(
            f"File size is very small ({file_size} bytes). "
            "Legitimate passport scans are usually larger."
        )

    if width < 200 or height < 200:
        obs.append(
            f"Image dimensions ({width}×{height} px) are unusually small for a passport scan."
        )

    # Date inconsistency: creation vs modification
    dt_original = structured_exif.get("datetime_original")
    dt_main     = structured_exif.get("datetime")
    if dt_original and dt_main and dt_original != dt_main:
        obs.append(
            f"DateTime ({dt_main}) differs from DateTimeOriginal ({dt_original}). "
            "This may indicate post-capture editing."
        )

    return obs


# ── Public API ────────────────────────────────────────────────────────────────

def run_metadata_analysis(
    source: Union[str, Path, bytes],
) -> Dict[str, Any]:
    """
    Extract image metadata and generate rule-based observations.

    Args:
        source: File path (str/Path) or raw image bytes.

    Returns:
        Dict with keys:
            success        (bool)
            image_info     (dict)  — format, size, mode, file_size
            exif_raw       (dict)  — all readable EXIF tags
            exif_fields    (dict)  — curated subset of EXIF fields
            observations   (list)  — cautious rule-based notes
            error          (str | None)

    NOTE: observations are informational only and do NOT constitute
    a determination of tampering or fraud.
    """
    result: Dict[str, Any] = {
        "success":    False,
        "image_info": {},
        "exif_raw":   {},
        "exif_fields": {},
        "observations": [],
        "error":      None,
    }

    # ── Load ──────────────────────────────────────────────────────────────────
    file_size: Optional[int] = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            result["error"] = f"Image not found: {path}"
            return result
        try:
            file_size = path.stat().st_size
            img = Image.open(path)
            img.load()
        except UnidentifiedImageError:
            result["error"] = f"Cannot identify image file: {path}"
            return result
        except Exception as exc:
            result["error"] = f"Failed to open '{path}': {exc}"
            return result
    elif isinstance(source, (bytes, bytearray)):
        file_size = len(source)
        try:
            img = Image.open(io.BytesIO(source))
            img.load()
        except Exception as exc:
            result["error"] = f"Cannot decode image bytes: {exc}"
            return result
    else:
        result["error"] = f"source must be a path or bytes, got {type(source)}"
        return result

    # ── Basic image info ──────────────────────────────────────────────────────
    fmt = (img.format or "UNKNOWN").upper()
    width, height = img.size

    result["image_info"] = {
        "format":    fmt,
        "width":     width,
        "height":    height,
        "mode":      img.mode,
        "file_size": file_size,
    }

    # ── EXIF ──────────────────────────────────────────────────────────────────
    raw_exif = _extract_exif(img)
    structured_exif = _build_structured_exif(raw_exif)

    result["exif_raw"]    = raw_exif
    result["exif_fields"] = structured_exif

    # ── Observations ──────────────────────────────────────────────────────────
    result["observations"] = _generate_observations(
        fmt=fmt,
        structured_exif=structured_exif,
        raw_exif=raw_exif,
        file_size=file_size,
        width=width,
        height=height,
    )

    result["success"] = True
    return result
