"""
Image preprocessing module using OpenCV and Pillow.

Provides functions to load an image, upscale if small, convert to grayscale,
and enhance contrast to maximize OCR accuracy on passport documents.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def load_image(image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Image.Image:
    """Load an image input into a PIL Image in RGB mode."""
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            return Image.fromarray(rgb)
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def resize_to_max(img: Image.Image, max_px: int = 2048, min_size: int = 1600) -> Image.Image:
    """
    Do not downscale images if reasonably sized.
    Only resize down if exceeds max_px or resize up (1.5-2x) if the image is small (<1600px),
    which helps EasyOCR read small visual-zone text.
    """
    w, h = img.size
    longest = max(w, h)
    if longest == 0:
        return img
    if longest > max_px:
        scale = max_px / longest
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif longest < min_size:
        scale = min_size / longest
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert PIL Image to grayscale (L mode)."""
    return img.convert("L")


def enhance_contrast(img: Image.Image, factor: float = 1.6) -> Image.Image:
    """Enhance image contrast using PIL and CLAHE."""
    enhanced_pil = ImageEnhance.Contrast(img).enhance(factor)
    gray_np = np.array(enhanced_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_np = clahe.apply(gray_np)
    return Image.fromarray(clahe_np).convert("RGB")


def enhance_sharpness(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Enhance image sharpness using PIL."""
    return ImageEnhance.Sharpness(img).enhance(factor)


def deskew(arr: np.ndarray) -> np.ndarray:
    """Deskew image using OpenCV minAreaRect if rotation angle is detected."""
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.copy()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return arr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return arr
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    max_size: int = 2048,
    contrast_factor: float = 1.5,
    sharpen_factor: float = 2.0,
    do_deskew: bool = True,
) -> Image.Image:
    """
    Full preprocessing pipeline: load -> upscale if small -> enhance contrast -> sharpen -> deskew.
    """
    img = load_image(image_input)
    img = resize_to_max(img, max_px=max_size, min_size=1600)
    img = enhance_contrast(img, contrast_factor)
    img = enhance_sharpness(img, sharpen_factor)
    if do_deskew:
        arr = np.array(img)
        arr = deskew(arr)
        img = Image.fromarray(arr).convert("RGB")
    return img.convert("RGB")


def preprocess_for_field(img: Image.Image, contrast_factor: float = 1.8, sharpen_factor: float = 2.5) -> Image.Image:
    """Preprocess a cropped field region for OCR."""
    img = load_image(img)
    img = resize_to_max(img, max_px=1024, min_size=1200)
    img = enhance_contrast(img, contrast_factor)
    img = enhance_sharpness(img, sharpen_factor)
    return img.convert("RGB")


def preprocess_for_mrz(img: Image.Image) -> Image.Image:
    """Preprocess cropped MRZ strip."""
    gray = img.convert("L")
    w, h = gray.size
    if h < 100:
        scale = 100 / h
        gray = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    arr = np.array(gray)
    arr = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
    arr = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(arr).convert("L")


def crop_region(
    img: Image.Image,
    box: tuple[float, float, float, float],
    relative: bool = True,
) -> Image.Image:
    """Crop a sub-region of an image."""
    w, h = img.size
    if relative:
        l, u, r, lo = box
        box = (int(l * w), int(u * h), int(r * w), int(lo * h))
    return img.crop(box)
