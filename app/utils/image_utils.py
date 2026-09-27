"""
Image utility helpers for loading, validating, and encoding images.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "BMP", "TIFF", "WEBP"}


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load a PIL Image from raw bytes.

    Args:
        data: Raw image bytes (from file upload or HTTP request).

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the bytes cannot be decoded as a supported image.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # detect truncated files early
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Re-open after verify (verify closes the file pointer)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def validate_image_format(img: Image.Image) -> Tuple[bool, str]:
    """
    Check that the image is in a supported format and within size limits.

    Args:
        img: PIL Image object.

    Returns:
        (is_valid, message) tuple.
    """
    fmt = (img.format or "UNKNOWN").upper()
    if fmt not in SUPPORTED_FORMATS and fmt != "UNKNOWN":
        return False, f"Unsupported format: {fmt}. Supported: {SUPPORTED_FORMATS}"

    w, h = img.size
    max_px = settings.OCR_MAX_IMAGE_SIZE
    if max(w, h) > max_px * 2:
        return False, f"Image too large: {w}x{h}. Maximum {max_px*2}px on longest edge."

    if w < 100 or h < 100:
        return False, f"Image too small: {w}x{h}. Minimum 100x100px."

    return True, "OK"


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to NumPy array (RGB uint8)."""
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert NumPy array (RGB uint8) to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def encode_image_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL Image as a base64 string for API transport.

    Args:
        img: PIL Image.
        fmt: Output format (PNG / JPEG).

    Returns:
        Base64-encoded string.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image_from_path(path: str | Path) -> Image.Image:
    """
    Load a PIL Image from a filesystem path.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    with open(p, "rb") as fh:
        return load_image_from_bytes(fh.read())
