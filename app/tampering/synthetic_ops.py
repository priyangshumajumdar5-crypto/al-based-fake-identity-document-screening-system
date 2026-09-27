"""
Synthetic Tampering Operations — Module 3, Phase 2.

Provides individual, reproducible tampering transformations applied to
existing passport source images. Every function:
  - Accepts and returns PIL Images (never modifies source on disk).
  - Is seeded for reproducibility.
  - Returns a result dict documenting what was done.
  - Reports limitations instead of silently producing invalid outputs.

Transformations implemented:
  A. Photo replacement  — paste a portrait crop from a donor image.
  B. Text-region blot   — cover a text region with a solid fill (simulates
                          overwriting a field; realistic text rendering is
                          NOT claimed — requires a realistic font pipeline).
  C. Region copy-paste  — copy a rectangle from a donor location and paste.
  D. Local colour shift — apply a hue/brightness shift to a bounded region
                          (simulates localised editing without inpainting).

Dependencies: Pillow, NumPy (already in requirements.txt).
NO new packages are required.

NOTE: These are synthetic research-only transformations. They do not
      reproduce real passport security features or authentication marks.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError

# ── Type aliases ──────────────────────────────────────────────────────────────
Region = Tuple[int, int, int, int]   # (left, top, right, bottom)
PIL_IMG = Image.Image

# ── Supported extensions ───────────────────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_rgb(path: Union[str, Path]) -> PIL_IMG:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source image not found: {p}")
    if p.suffix.lower() not in _IMG_EXTS:
        raise ValueError(f"Unsupported format: {p.suffix}")
    try:
        img = Image.open(p)
        img.load()
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image: {p}")
    return img.convert("RGB")


def _safe_region(region: Region, img_w: int, img_h: int) -> Region:
    """Clamp a region to image bounds."""
    l, t, r, b = region
    l = max(0, min(l, img_w - 1))
    t = max(0, min(t, img_h - 1))
    r = max(l + 1, min(r, img_w))
    b = max(t + 1, min(b, img_h))
    return l, t, r, b


def _result_template(op: str) -> Dict[str, Any]:
    return {
        "success":          False,
        "operation":        op,
        "output_image":     None,   # PIL Image
        "affected_region":  None,
        "description":      "",
        "warnings":         [],
        "error":            None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# A. Photo Replacement
# ══════════════════════════════════════════════════════════════════════════════

# Default photo box for a typical Indian passport scan at ~900×640 px.
# Adjust per image set via the `photo_region` argument.
_DEFAULT_PHOTO_REGION: Region = (30, 115, 200, 295)


def tamper_photo_replacement(
    target_path:   Union[str, Path],
    donor_path:    Union[str, Path],
    photo_region:  Optional[Region] = None,
    seed:          int = 42,
) -> Dict[str, Any]:
    """
    Replace the portrait region of target with a crop from donor.

    Args:
        target_path:  Source passport image to be tampered.
        donor_path:   Different passport image supplying the replacement portrait.
        photo_region: (left, top, right, bottom) of portrait area in target.
                      Defaults to _DEFAULT_PHOTO_REGION.
        seed:         RNG seed (for reproducibility; not used in this op but
                      kept uniform across the API).

    Returns:
        result dict with output_image (PIL.Image) and metadata.
    """
    res = _result_template("photo_replacement")
    region = photo_region or _DEFAULT_PHOTO_REGION

    try:
        target = _load_rgb(target_path)
        donor  = _load_rgb(donor_path)
    except (FileNotFoundError, ValueError) as exc:
        res["error"] = str(exc)
        return res

    tw, th = target.size
    region = _safe_region(region, tw, th)
    rw = region[2] - region[0]
    rh = region[3] - region[1]

    if rw < 10 or rh < 10:
        res["error"] = f"Photo region too small after clamping: {rw}×{rh} px."
        return res

    # Crop portrait from donor (same relative box, or scale to donor dims)
    dw, dh = donor.size
    donor_region = _safe_region(region, dw, dh)
    donor_crop = donor.crop(donor_region)

    # Resize to exactly the target region size
    replacement = donor_crop.resize((rw, rh), Image.LANCZOS)

    # Paste onto a copy of target
    output = target.copy()
    output.paste(replacement, (region[0], region[1]))

    res["success"]         = True
    res["output_image"]    = output
    res["affected_region"] = region
    res["description"]     = (
        f"Portrait region {region} replaced with crop from donor image. "
        "Donor portrait origin recorded."
    )
    return res


# ══════════════════════════════════════════════════════════════════════════════
# B. Text Region Blot (field overwrite simulation)
# ══════════════════════════════════════════════════════════════════════════════

# Named regions for a typical 900×640 synthetic/real Indian passport scan.
# Coordinates are approximate; override via `region` argument.
TEXT_REGIONS: Dict[str, Region] = {
    "date_of_birth":  (220, 240, 450, 265),
    "date_of_expiry": (280, 318, 510, 345),
    "passport_number":(550, 70,  760, 100),
    "surname":        (220, 120, 550, 150),
    "given_names":    (220, 160, 700, 190),
    "date_of_issue":  (30,  318, 260, 345),
}


def tamper_text_region(
    target_path:      Union[str, Path],
    region_name:      Optional[str] = None,
    region:           Optional[Region] = None,
    fill_colour:      Tuple[int, int, int] = (245, 242, 235),  # matches parchment BG
    replacement_text: Optional[str] = None,
    seed:             int = 42,
) -> Dict[str, Any]:
    """
    Overwrite a named text region with a solid fill (and optional fake text).

    NOTE: Realistic font reconstruction is NOT implemented. If replacement_text
    is supplied it is drawn in the default PIL font. This is intentionally
    crude — it is a synthetic research transformation, not a realistic forgery.

    Args:
        target_path:      Source passport image.
        region_name:      Key in TEXT_REGIONS; used if `region` is None.
        region:           Explicit (left, top, right, bottom) override.
        fill_colour:      Background fill colour for the blotted area.
        replacement_text: If set, crude text drawn over fill. None = blank fill.
        seed:             RNG seed.

    Returns:
        result dict with output_image (PIL.Image) and metadata.
    """
    res = _result_template("text_region_blot")
    rng = random.Random(seed)

    # Resolve region
    if region is None and region_name is None:
        res["error"] = "Provide either region_name or region."
        return res
    if region is None:
        if region_name not in TEXT_REGIONS:
            res["error"] = (
                f"Unknown region_name '{region_name}'. "
                f"Available: {list(TEXT_REGIONS.keys())}"
            )
            return res
        region = TEXT_REGIONS[region_name]

    try:
        target = _load_rgb(target_path)
    except (FileNotFoundError, ValueError) as exc:
        res["error"] = str(exc)
        return res

    tw, th = target.size
    region = _safe_region(region, tw, th)
    rw, rh = region[2] - region[0], region[3] - region[1]

    if rw < 4 or rh < 4:
        res["error"] = f"Region too small after clamping: {rw}×{rh} px."
        return res

    output = target.copy()
    draw   = ImageDraw.Draw(output)
    draw.rectangle([region[0], region[1], region[2] - 1, region[3] - 1], fill=fill_colour)

    if replacement_text:
        # Draw in default PIL font — intentionally low-fidelity
        draw.text((region[0] + 2, region[1] + 2), replacement_text, fill=(20, 20, 60))

    res["success"]         = True
    res["output_image"]    = output
    res["affected_region"] = region
    res["description"]     = (
        f"Text region {region_name or 'custom'} {region} filled with solid colour. "
        + (f"Crude replacement text drawn (NOT realistic)." if replacement_text else "Blank fill.")
    )
    if replacement_text:
        res["warnings"].append(
            "Replacement text uses PIL default font — not visually realistic. "
            "This is a known limitation of the synthetic approach."
        )
    return res


# ══════════════════════════════════════════════════════════════════════════════
# C. Region Copy-Paste
# ══════════════════════════════════════════════════════════════════════════════

def tamper_region_copypaste(
    target_path:   Union[str, Path],
    donor_path:    Union[str, Path],
    source_region: Region,
    paste_region:  Region,
    blend_alpha:   float = 1.0,
    seed:          int   = 42,
) -> Dict[str, Any]:
    """
    Copy source_region from donor and paste into paste_region of target.

    Args:
        target_path:   Destination image.
        donor_path:    Image to copy from (may be the same file as target
                       for self-copy-paste, which is valid).
        source_region: (l, t, r, b) region to copy from donor.
        paste_region:  (l, t, r, b) region to paste into target.
                       Donor crop is resized to fit paste_region.
        blend_alpha:   0.0 = fully transparent (no effect) … 1.0 = fully opaque.
        seed:          RNG seed.

    Returns:
        result dict with output_image (PIL.Image) and metadata.
    """
    res = _result_template("region_copypaste")

    try:
        target = _load_rgb(target_path)
        donor  = _load_rgb(donor_path)
    except (FileNotFoundError, ValueError) as exc:
        res["error"] = str(exc)
        return res

    tw, th = target.size
    dw, dh = donor.size

    src = _safe_region(source_region, dw, dh)
    dst = _safe_region(paste_region,  tw, th)

    crop = donor.crop(src)
    paste_w = dst[2] - dst[0]
    paste_h = dst[3] - dst[1]

    if paste_w < 1 or paste_h < 1:
        res["error"] = f"Paste region is empty after clamping: {dst}"
        return res

    crop_resized = crop.resize((paste_w, paste_h), Image.LANCZOS)
    output = target.copy()

    if blend_alpha >= 1.0:
        output.paste(crop_resized, (dst[0], dst[1]))
    else:
        # Alpha blend
        bg = output.crop(dst)
        blended = Image.blend(bg, crop_resized, alpha=blend_alpha)
        output.paste(blended, (dst[0], dst[1]))

    res["success"]         = True
    res["output_image"]    = output
    res["affected_region"] = dst
    res["description"]     = (
        f"Crop {src} from donor pasted at {dst} in target (alpha={blend_alpha:.2f})."
    )
    return res


# ══════════════════════════════════════════════════════════════════════════════
# D. Local Colour Shift
# ══════════════════════════════════════════════════════════════════════════════

def tamper_local_colour_shift(
    target_path:     Union[str, Path],
    region:          Region,
    brightness_delta: float = 30.0,   # added to all channels in region
    hue_shift_deg:    float = 0.0,    # degrees (0 = no hue change)
    seed:             int   = 42,
) -> Dict[str, Any]:
    """
    Apply a brightness/colour shift to a bounded region.

    Args:
        target_path:       Source passport image.
        region:            (left, top, right, bottom) of region to shift.
        brightness_delta:  Value added to each RGB channel (can be negative).
        hue_shift_deg:     Hue rotation in degrees. Requires numpy.
        seed:              RNG seed.

    Returns:
        result dict with output_image (PIL.Image) and metadata.
    """
    res = _result_template("local_colour_shift")

    if brightness_delta == 0 and hue_shift_deg == 0:
        res["error"] = "Both brightness_delta and hue_shift_deg are 0 — no transformation."
        return res

    try:
        target = _load_rgb(target_path)
    except (FileNotFoundError, ValueError) as exc:
        res["error"] = str(exc)
        return res

    tw, th = target.size
    region = _safe_region(region, tw, th)
    rw, rh = region[2] - region[0], region[3] - region[1]

    if rw < 2 or rh < 2:
        res["error"] = f"Region too small: {rw}×{rh} px."
        return res

    output = target.copy()
    patch  = np.array(output.crop(region), dtype=np.float32)

    if brightness_delta != 0:
        patch = np.clip(patch + brightness_delta, 0, 255)

    if hue_shift_deg != 0:
        # HSV hue rotation using numpy (no cv2 dependency)
        patch_uint8 = patch.astype(np.uint8)
        patch_pil   = Image.fromarray(patch_uint8).convert("HSV")
        hsv         = np.array(patch_pil, dtype=np.float32)
        hsv[..., 0] = (hsv[..., 0] + (hue_shift_deg / 360.0) * 255) % 255
        patch = np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"),
                         dtype=np.float32)

    patch_img = Image.fromarray(patch.clip(0, 255).astype(np.uint8))
    output.paste(patch_img, (region[0], region[1]))

    res["success"]         = True
    res["output_image"]    = output
    res["affected_region"] = region
    res["description"]     = (
        f"Local colour shift at {region}: "
        f"brightness_delta={brightness_delta}, hue_shift_deg={hue_shift_deg}."
    )
    return res



# ══════════════════════════════════════════════════════════════════════════════
# E. Pen overwrite / scribble (handwritten-style edit, like real forgeries)
# ══════════════════════════════════════════════════════════════════════════════

def _get_font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tamper_pen_overwrite(
    target_path: Union[str, Path],
    region:      Region,
    text:        Optional[str] = None,
    ink:         Tuple[int, int, int] = (30, 40, 130),
    seed:        int = 42,
) -> Dict[str, Any]:
    """
    Draw pen-style strike-through strokes (and optional handwritten-looking text)
    over a region. text=None -> scribble only (e.g. scribbling out the MRZ).
    Approximate simulation of hand-edited passports; not a real forgery.
    """
    res = _result_template("pen_overwrite")
    rng = random.Random(seed)

    try:
        target = _load_rgb(target_path)
    except (FileNotFoundError, ValueError) as exc:
        res["error"] = str(exc)
        return res

    tw, th = target.size
    l, t, r, b = _safe_region(region, tw, th)
    rw, rh = r - l, b - t
    if rw < 8 or rh < 8:
        res["error"] = f"Region too small: {rw}x{rh} px."
        return res

    col = tuple(int(max(0, min(255, c + rng.randint(-10, 10)))) for c in ink) + (230,)
    layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    n_strokes = 1 if text else rng.randint(2, 4)
    for _ in range(n_strokes):
        y = rng.uniform(0.25 * rh, 0.75 * rh)
        pts, x = [], 0
        while x < rw:
            pts.append((x, y + rng.uniform(-0.18 * rh, 0.18 * rh)))
            x += max(4, rw // 24)
        pts.append((rw, y))
        d.line(pts, fill=col, width=max(2, rh // 10), joint="curve")

    if text:
        d.text((rng.randint(0, max(1, rw // 10)), 0), text, font=_get_font(int(rh * 0.85)), fill=col)
        layer = layer.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC)

    layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    out = target.copy()
    out.paste(layer, (l, t), layer)

    res["success"] = True
    res["output_image"] = out
    res["affected_region"] = (l, t, r, b)
    res["description"] = f"Pen-style overlay at {(l, t, r, b)}" + (f" with text '{text}'." if text else " (scribble only).")
    res["warnings"].append("Synthetic pen overlay — approximate simulation, not a real forgery.")
    return res


    