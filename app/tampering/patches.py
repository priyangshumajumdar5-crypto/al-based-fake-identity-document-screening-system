"""Patch utilities — Module 3 (patch-based tampering classifier)."""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

PATCH_SIZE = 224
NORM_WIDTH = 768            # every image is rescaled to this width before patching
POS_MIN_COVERAGE_RATIO = 0.05 # patch is "tampered" if >= 5% of its pixels were edited (configurable threshold)
POS_MIN_FRAC = POS_MIN_COVERAGE_RATIO
MASK_DIFF_THRESHOLD = 10    # pixel difference that counts as "edited"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Decision bands for the aggregated tampered score (single source of truth)
BAND_GENUINE_MAX = 0.35
BAND_TAMPERED_MIN = 0.65


def score_to_band(score: float) -> str:
    if score < BAND_GENUINE_MAX:
        return "GENUINE"
    if score > BAND_TAMPERED_MIN:
        return "TAMPERED"
    return "UNCERTAIN"


# ── Loading / sizing ──────────────────────────────────────────────────────────

def normalise_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == NORM_WIDTH:
        return img
    return img.resize((NORM_WIDTH, max(1, round(h * NORM_WIDTH / w))), Image.LANCZOS)


def load_normalised(path: Union[str, Path]) -> Image.Image:
    return normalise_size(Image.open(path).convert("RGB"))


def pad_to_min(arr: np.ndarray, mask: Optional[np.ndarray] = None):
    """Pad (edge) so both sides are >= PATCH_SIZE."""
    h, w = arr.shape[:2]
    ph, pw = max(0, PATCH_SIZE - h), max(0, PATCH_SIZE - w)
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
        if mask is not None:
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=False)
    return arr, mask


# ── Tamper mask (which pixels were edited) ────────────────────────────────────

def genuine_reference_path(tampered_path: Union[str, Path]) -> Optional[Path]:
    """.../images/tampered/<doc>/<file>.jpg -> .../images/genuine/<doc>/<doc>_genuine_original.jpg"""
    p = Path(tampered_path)
    doc = p.parent.name
    cand = p.parent.parent.parent / "genuine" / doc / f"{doc}_genuine_original.jpg"
    return cand if cand.exists() else None


def tamper_mask(tampered_img: Image.Image, genuine_path: Union[str, Path]) -> np.ndarray:
    """Boolean HxW mask: True where the tampered image differs from its genuine original."""
    g = load_normalised(genuine_path)
    if g.size != tampered_img.size:
        g = g.resize(tampered_img.size, Image.LANCZOS)
    a = np.asarray(tampered_img, dtype=np.int16)
    b = np.asarray(g, dtype=np.int16)
    return np.abs(a - b).max(axis=2) > MASK_DIFF_THRESHOLD


def mask_from_affected_region(
    img_height: int,
    img_width: int,
    region: Tuple[int, int, int, int],
    orig_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Construct a boolean HxW mask from a bounding box region (l, t, r, b).
    Scales region coordinates if orig_size (orig_w, orig_h) is provided.
    """
    mask = np.zeros((img_height, img_width), dtype=bool)
    if not region or len(region) != 4:
        return mask

    l, t, r, b = region
    if orig_size and orig_size[0] > 0 and orig_size[1] > 0:
        orig_w, orig_h = orig_size
        sx = img_width / orig_w
        sy = img_height / orig_h
        l, t, r, b = int(l * sx), int(t * sy), int(r * sx), int(b * sy)

    l = max(0, min(l, img_width - 1))
    t = max(0, min(t, img_height - 1))
    r = max(l + 1, min(r, img_width))
    b = max(t + 1, min(b, img_height))

    mask[t:b, l:r] = True
    return mask


# ── Patch sampling (training) ─────────────────────────────────────────────────

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def sample_patch(
    arr: np.ndarray,
    mask: Optional[np.ndarray],
    want_tampered: bool,
    rng: random.Random,
) -> Tuple[np.ndarray, int]:
    """Return (patch uint8 PxPx3, label). mask=None means genuine image (always label 0)."""
    S = PATCH_SIZE
    arr, mask = pad_to_min(arr, mask)
    h, w = arr.shape[:2]

    if want_tampered and mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        jit = S // 4
        for _ in range(10):
            j = rng.randrange(len(ys))
            y0 = _clamp(int(ys[j]) - S // 2 + rng.randint(-jit, jit), 0, h - S)
            x0 = _clamp(int(xs[j]) - S // 2 + rng.randint(-jit, jit), 0, w - S)
            if mask[y0:y0 + S, x0:x0 + S].mean() >= POS_MIN_FRAC:
                return arr[y0:y0 + S, x0:x0 + S], 1

    last = None
    for _ in range(30):
        y0 = rng.randint(0, h - S)
        x0 = rng.randint(0, w - S)
        patch = arr[y0:y0 + S, x0:x0 + S]
        frac = 0.0 if mask is None else float(mask[y0:y0 + S, x0:x0 + S].mean())
        if frac == 0.0:
            return patch, 0
        last = (patch, frac)
    return last[0], (1 if last[1] >= POS_MIN_FRAC else 0)


def degrade(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random phone-photo style degradation (applied to BOTH classes)."""
    H, W = patch.shape[:2]
    img = Image.fromarray(patch)
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 0.9)
        small = img.resize((max(8, int(W * s)), max(8, int(H * s))), Image.BILINEAR)
        img = small.resize((W, H), Image.BILINEAR)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.6:
        arr = (arr - 128.0) * rng.uniform(0.85, 1.15) + 128.0 + rng.uniform(-15, 15)
    if rng.random() < 0.4:
        noise = np.random.RandomState(rng.randrange(2 ** 31)).normal(0, rng.uniform(1, 5), arr.shape)
        arr = arr + noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.8:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(45, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return np.asarray(img)


# ── Tiling + tensor conversion (inference) ────────────────────────────────────

def tile_boxes(h: int, w: int, size: int = PATCH_SIZE, stride: int = 112):
    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    if ys[-1] != max(h - size, 0):
        ys.append(max(h - size, 0))
    if xs[-1] != max(w - size, 0):
        xs.append(max(w - size, 0))
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def patch_to_tensor(patch: np.ndarray):
    import torch
    x = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std