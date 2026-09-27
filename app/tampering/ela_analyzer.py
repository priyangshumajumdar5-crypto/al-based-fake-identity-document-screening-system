"""
ELA Analyzer — Module 3, Phase 1.

Error Level Analysis: highlights compression inconsistencies that can
indicate localised image edits. ELA is a forensic *visualisation tool*;
it does NOT prove tampering on its own.

Dependencies: Pillow, NumPy  (both already in requirements.txt)
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

# ── Constants ─────────────────────────────────────────────────────────────────
_SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_DEFAULT_QUALITY = 90          # JPEG recompression quality for ELA
_DEFAULT_SCALE   = 15          # Amplification factor for difference visualisation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(source: Union[str, Path, bytes]) -> Tuple[Image.Image, str]:
    """
    Load and return an RGB PIL Image plus a format note.

    Args:
        source: File path (str or Path) or raw image bytes.

    Returns:
        (image_rgb, format_note)

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be decoded.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(_SUPPORTED_FORMATS)}"
            )
        try:
            img = Image.open(path)
            img.load()          # force decode
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file: {path}")
        except Exception as exc:
            raise ValueError(f"Failed to open image '{path}': {exc}") from exc
        original_format = (img.format or path.suffix.lstrip(".").upper()).upper()
    elif isinstance(source, (bytes, bytearray)):
        try:
            img = Image.open(io.BytesIO(source))
            img.load()
        except Exception as exc:
            raise ValueError(f"Cannot decode image bytes: {exc}") from exc
        original_format = (img.format or "UNKNOWN").upper()
    else:
        raise TypeError(f"source must be a path or bytes, got {type(source)}")

    img_rgb = img.convert("RGB")
    return img_rgb, original_format


def _recompress_to_jpeg(img_rgb: Image.Image, quality: int) -> Image.Image:
    """
    Recompress an RGB image to JPEG at the given quality and reload it.
    Uses an in-memory buffer to avoid touching the filesystem when possible.
    """
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    recompressed.load()
    return recompressed.convert("RGB")


# ── Public API ────────────────────────────────────────────────────────────────

def run_ela(
    source: Union[str, Path, bytes],
    quality: int = _DEFAULT_QUALITY,
    scale: int = _DEFAULT_SCALE,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run Error Level Analysis on an image.

    Args:
        source:    File path (str/Path) or raw image bytes.
        quality:   JPEG recompression quality (1-95). Default 90.
        scale:     Amplification applied to the pixel-difference image
                   to improve visual contrast. Default 15.
        save_path: Optional path to save the ELA visualisation PNG.
                   Must differ from the source path.

    Returns:
        Dict with keys:
            success       (bool)
            ela_image     (PIL.Image.Image | None)  — amplified difference image
            ela_array     (np.ndarray | None)       — uint8 RGB array (H×W×3)
            mean_ela      (float | None)             — mean pixel difference (pre-scale)
            max_ela       (float | None)             — max pixel difference (pre-scale)
            quality_used  (int)
            scale_used    (int)
            source_format (str)
            converted     (bool)   — True when source was converted to JPEG for ELA
            save_path     (str | None)
            error         (str | None)
    """
    result: Dict[str, Any] = {
        "success":       False,
        "ela_image":     None,
        "ela_array":     None,
        "mean_ela":      None,
        "max_ela":       None,
        "quality_used":  quality,
        "scale_used":    scale,
        "source_format": None,
        "converted":     False,
        "save_path":     None,
        "error":         None,
    }

    # Validate quality
    if not (1 <= quality <= 95):
        result["error"] = f"quality must be 1-95, got {quality}"
        return result

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        img_rgb, source_format = _load_image(source)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
        return result

    result["source_format"] = source_format

    # For non-JPEG sources, note the conversion
    if source_format not in ("JPEG", "JPG"):
        result["converted"] = True

    # ── Recompress & diff ─────────────────────────────────────────────────────
    try:
        recompressed = _recompress_to_jpeg(img_rgb, quality)
    except Exception as exc:
        result["error"] = f"Recompression failed: {exc}"
        return result

    orig_arr = np.array(img_rgb, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr)          # float32, range 0-255

    result["mean_ela"] = float(np.mean(diff))
    result["max_ela"]  = float(np.max(diff))

    # Amplify for visualisation and clip to uint8
    # Dynamic display contrast enhancement so ELA images remain visually readable
    # without altering the raw mean_ela and max_ela measurements.
    max_diff = result["max_ela"]
    if max_diff > 0:
        eff_scale = max(float(scale), min(40.0, 255.0 / (max_diff + 1e-5)))
    else:
        eff_scale = float(scale)

    amplified = np.clip(diff * eff_scale, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(amplified, mode="RGB")

    result["ela_image"] = ela_img
    result["ela_array"] = amplified

    # ── Optional save ─────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        # Guard: do not overwrite the source
        if isinstance(source, (str, Path)) and save_path.resolve() == Path(source).resolve():
            result["error"] = (
                "save_path must differ from the source image path. "
                "ELA visualisation was NOT saved to avoid overwriting the original."
            )
            # Still return the ELA result even though save failed
            result["success"] = True
            return result
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ela_img.save(str(save_path))
            result["save_path"] = str(save_path)
        except Exception as exc:
            result["error"] = f"Failed to save ELA image to '{save_path}': {exc}"
            result["success"] = True   # ELA itself succeeded
            return result

    result["success"] = True
    result["interpretation"] = interpret_ela_metrics(result["mean_ela"], result["max_ela"])
    return result


def interpret_ela_metrics(mean_ela: float, max_ela: float) -> Dict[str, Any]:
    """
    Translate technical ELA values into simple, threshold-based ranges
    for ordinary users.
    """
    if mean_ela < 5.0:
        status = "LIKELY_GENUINE"
        label = "Consistent Compression (Likely Genuine)"
        description = (
            f"Low Error Level variation (Mean ELA: {mean_ela:.1f}). "
            "Compression levels across the document are uniform, suggesting the image has not been selectively edited."
        )
    elif mean_ela <= 12.0:
        status = "INCONCLUSIVE"
        label = "Moderate Compression Variance (Normal Save Artifacts)"
        description = (
            f"Moderate Error Level variation (Mean ELA: {mean_ela:.1f}). "
            "Typical of standard multi-save compression or normal scan artifacts. Document appears consistent."
        )
    else:
        status = "POTENTIALLY_TAMPERED"
        label = "High Compression Anomaly (Potential Local Edits)"
        description = (
            f"Elevated Error Level variation (Mean ELA: {mean_ela:.1f}, Max ELA: {max_ela:.1f}). "
            "Specific areas exhibit significant re-compression differences, suggesting localized editing or splicing."
        )

    return {
        "status": status,
        "label": label,
        "description": description,
        "mean_ela": round(mean_ela, 2),
        "max_ela": round(max_ela, 2),
    }
