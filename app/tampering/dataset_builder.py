"""
Dataset Builder — Module 3, Phase 2.

Generates the labeled synthetic tampering dataset from existing passport
source images and writes it to disk in a layout fully compatible with the
Phase 3 dataset_loader.py interface.

Dataset root:  data/tampering_dataset/
  images/
    genuine/    <base_doc_id>/<sample_id>.jpg
    tampered/   <base_doc_id>/<sample_id>.jpg
  manifests/
    dataset_manifest.csv
  reports/
    generation_summary.json

Phase 3 loader expects manifest.csv at <dataset_root>/manifest.csv with
columns: image_path, label, base_document_id.

This builder writes a richer manifest to manifests/dataset_manifest.csv AND
a minimal Phase-3-compatible copy to manifest.csv at the root.

IMPORTANT — Label basis:
  * Genuine: original unmodified passport source image. Authenticity of the
    real-world passport is UNKNOWN — the source image is used as the
    "genuine reference" for the synthetic dataset only.
  * Tampered: a synthetic variant produced by a documented transformation.
  * Do NOT use ELA scores to assign labels.
"""
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from app.tampering.synthetic_ops import (
    tamper_photo_replacement,
    tamper_text_region,
    tamper_region_copypaste,
    tamper_local_colour_shift,
    tamper_pen_overwrite,
    TEXT_REGIONS,
    _DEFAULT_PHOTO_REGION,
)

# ── Labels — must match Phase 3 ───────────────────────────────────────────────
LABEL_GENUINE  = 0
LABEL_TAMPERED = 1

# ── Extended manifest columns ─────────────────────────────────────────────────
_MANIFEST_FIELDS = [
    "image_path",        # relative from dataset_root (Phase 3 required)
    "label",             # 0 / 1       (Phase 3 required)
    "base_document_id",  # e.g. passport1 (Phase 3 required)
    "sample_id",
    "class_name",
    "source_image",
    "is_synthetic",
    "tampering_type",
    "affected_region",
    "generation_seed",
    "label_basis",
    "warnings",
]

# ── Phase 3 minimal columns (root manifest.csv) ───────────────────────────────
_P3_FIELDS = ["image_path", "label", "base_document_id", "affected_region"]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Source discovery ──────────────────────────────────────────────────────────

def discover_source_images(source_dir: Union[str, Path]) -> List[Path]:
    """
    Find all image files under source_dir. Returns sorted list of Path objects.
    Does NOT filter by assumed authenticity — label basis is documented per sample.
    """
    d = Path(source_dir)
    if not d.exists():
        raise FileNotFoundError(f"Source directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_image(img: Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    if fmt == "JPEG":
        img.save(str(path), format="JPEG", quality=quality)
    else:
        img.save(str(path), format="PNG")


# ── Genuine sample creation ───────────────────────────────────────────────────

def create_genuine_sample(
    source_path:  Path,
    output_dir:   Path,
    base_doc_id:  str,
) -> Dict[str, Any]:
    """
    Copy source image as a genuine sample.

    NOTE: The source image is used as the "genuine reference" for the
    synthetic research dataset. Real-world authenticity is UNKNOWN.
    """
    sample_id = f"{base_doc_id}_genuine_original"
    out_path  = output_dir / "images" / "genuine" / base_doc_id / f"{sample_id}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(source_path).convert("RGB")
    _save_image(img, out_path)

    return {
        "image_path":       str(out_path),
        "label":            LABEL_GENUINE,
        "base_document_id": base_doc_id,
        "sample_id":        sample_id,
        "class_name":       "GENUINE",
        "source_image":     source_path.name,
        "is_synthetic":     False,
        "tampering_type":   "none",
        "affected_region":  "",
        "generation_seed":  "",
        "label_basis":      (
            "Original source image used as genuine reference for synthetic dataset. "
            "Real-world authenticity is unknown — requires manual review."
        ),
        "warnings":         "Authenticity not confirmed — synthetic dataset reference only.",
        "success":          True,
        "error":            None,
    }


# ── Tampered sample creation ──────────────────────────────────────────────────

# Regions measured on a 768x540 passport scan; scaled to each image's own size.
_BASE_W, _BASE_H = 768, 540
_REL = {
    "photo":  (40, 110, 258, 335),
    "dob":    (270, 243, 420, 270),
    "number": (585, 95, 735, 122),
    "expiry": (585, 398, 735, 420),
    "mrz":    (35, 462, 745, 525),
}


def _scaled(key: str, iw: int, ih: int) -> Tuple[int, int, int, int]:
    l, t, r, b = _REL[key]
    sx, sy = iw / _BASE_W, ih / _BASE_H
    return int(l * sx), int(t * sy), int(r * sx), int(b * sy)


def create_tampered_samples(
    source_path:      Path,
    donor_path:       Path,
    output_dir:       Path,
    base_doc_id:      str,
    seed:             int = 42,
    photo_region:     Optional[Tuple[int,int,int,int]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate 7 tampered variants per source image (regions scaled to image size):
    photo swap, text blot, copy-paste, colour shift, pen DOB, pen passport number,
    MRZ scribble. Failed operations are recorded, not silently skipped.
    """
    records: List[Dict[str, Any]] = []
    base_out = output_dir / "images" / "tampered" / base_doc_id
    with Image.open(source_path) as _im:
        iw, ih = _im.size

    def R(key: str):
        return _scaled(key, iw, ih)

    jobs = [
        ("photo_repl", "photo_replacement", lambda: tamper_photo_replacement(
            source_path, donor_path, photo_region=photo_region or R("photo"), seed=seed)),
        ("text_dob", "text_region_blot", lambda: tamper_text_region(
            source_path, region=R("dob"), replacement_text="01/01/2000", seed=seed)),
        ("copypaste", "region_copypaste", lambda: tamper_region_copypaste(
            source_path, source_path,
            source_region=(0, int(ih * 0.68), iw, int(ih * 0.78)),
            paste_region=(0, int(ih * 0.40), iw, int(ih * 0.50)),
            blend_alpha=0.85, seed=seed)),
        ("colourshift", "local_colour_shift", lambda: tamper_local_colour_shift(
            source_path, region=R("expiry"), brightness_delta=40.0, hue_shift_deg=20.0, seed=seed)),
        ("pen_dob", "pen_overwrite", lambda: tamper_pen_overwrite(
            source_path, R("dob"), text="10/12/1988", seed=seed)),
        ("pen_number", "pen_overwrite", lambda: tamper_pen_overwrite(
            source_path, R("number"), text="X9654304", seed=seed)),
        ("mrz_scribble", "pen_scribble", lambda: tamper_pen_overwrite(
            source_path, R("mrz"), text=None, seed=seed)),
    ]

    for tag, op_name, run in jobs:
        sample_id = f"{base_doc_id}_tampered_{tag}_{seed}"
        out_path = base_out / f"{sample_id}.jpg"
        try:
            op = run()
            if op["success"]:
                _save_image(op["output_image"], out_path)
                records.append(_tampered_record(
                    out_path, source_path, base_doc_id, sample_id,
                    op_name, op["affected_region"], seed, op.get("warnings", []),
                ))
            else:
                records.append(_failed_record(base_doc_id, op_name, seed, op["error"]))
        except Exception as exc:
            records.append(_failed_record(base_doc_id, op_name, seed, str(exc)))

    return records


def _tampered_record(
    out_path:     Path,
    source_path:  Path,
    base_doc_id:  str,
    sample_id:    str,
    tampering_type: str,
    affected_region: Any,
    seed:         int,
    warnings:     List[str],
) -> Dict[str, Any]:
    return {
        "image_path":       str(out_path),
        "label":            LABEL_TAMPERED,
        "base_document_id": base_doc_id,
        "sample_id":        sample_id,
        "class_name":       "TAMPERED",
        "source_image":     source_path.name,
        "is_synthetic":     True,
        "tampering_type":   tampering_type,
        "affected_region":  str(affected_region),
        "generation_seed":  seed,
        "label_basis":      f"Synthetic variant generated by '{tampering_type}' operation.",
        "warnings":         "; ".join(warnings) if warnings else "",
        "success":          True,
        "error":            None,
    }


def _failed_record(
    base_doc_id: str, tampering_type: str, seed: int, error: str
) -> Dict[str, Any]:
    return {
        "image_path":       None,
        "label":            LABEL_TAMPERED,
        "base_document_id": base_doc_id,
        "sample_id":        f"{base_doc_id}_FAILED_{tampering_type}",
        "class_name":       "TAMPERED",
        "source_image":     "",
        "is_synthetic":     True,
        "tampering_type":   tampering_type,
        "affected_region":  "",
        "generation_seed":  seed,
        "label_basis":      "FAILED — not saved.",
        "warnings":         f"Generation failed: {error}",
        "success":          False,
        "error":            error,
    }


# ── Manifest writing ──────────────────────────────────────────────────────────

def write_manifests(
    records:      List[Dict[str, Any]],
    dataset_root: Path,
) -> Tuple[Path, Path]:
    """
    Write full manifest (manifests/dataset_manifest.csv) and
    Phase-3-compatible minimal manifest (manifest.csv at root).

    Skips failed records (image_path=None) from both manifests.

    Returns:
        (full_manifest_path, p3_manifest_path)
    """
    valid = [r for r in records if r.get("success") and r.get("image_path")]

    # Make image_path relative to dataset_root for portability
    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(dataset_root))
        except ValueError:
            return p

    # Full manifest
    full_dir = dataset_root / "manifests"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / "dataset_manifest.csv"

    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in valid:
            row = dict(r)
            row["image_path"] = rel(r["image_path"])
            writer.writerow(row)

    # Phase 3 minimal manifest at root
    p3_path = dataset_root / "manifest.csv"
    with open(p3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_P3_FIELDS)
        writer.writeheader()
        for r in valid:
            writer.writerow({
                "image_path":       rel(r["image_path"]),
                "label":            r["label"],
                "base_document_id": r["base_document_id"],
                "affected_region":  r.get("affected_region", ""),
            })

    return full_path, p3_path


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary(
    records:      List[Dict[str, Any]],
    dataset_root: Path,
    source_dir:   str,
) -> Path:
    """Write a JSON generation summary to reports/generation_summary.json."""
    reports_dir = dataset_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "generation_summary.json"

    successful = [r for r in records if r.get("success")]
    failed     = [r for r in records if not r.get("success")]
    genuine    = [r for r in successful if r["label"] == LABEL_GENUINE]
    tampered   = [r for r in successful if r["label"] == LABEL_TAMPERED]

    summary = {
        "phase":           "Phase 2 — Synthetic Tampering Dataset Generation",
        "source_dir":      str(source_dir),
        "dataset_root":    str(dataset_root),
        "total_records":   len(records),
        "successful":      len(successful),
        "failed":          len(failed),
        "genuine_count":   len(genuine),
        "tampered_count":  len(tampered),
        "tampering_types": list({r["tampering_type"] for r in tampered}),
        "failed_records":  [
            {"sample_id": r["sample_id"], "error": r["error"]} for r in failed
        ],
        "label_note": (
            "Genuine labels indicate original source images used as reference "
            "for this synthetic research dataset. Real-world passport authenticity "
            "is UNKNOWN and requires manual review. "
            "ELA scores were NOT used to assign labels."
        ),
        "synthetic_note": (
            "All tampered samples are synthetic research data generated by "
            "documented image transformations. They are NOT real forged documents."
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_path


# ── Dataset validation ────────────────────────────────────────────────────────

def validate_dataset(dataset_root: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate the generated dataset.

    Checks:
      - dataset_root exists
      - manifest.csv exists and is readable
      - all manifest image_path entries resolve to existing files
      - labels are valid (0 or 1)
      - no duplicate sample paths
      - class counts

    Returns:
        Dict with: valid, genuine_count, tampered_count, missing_files,
                   invalid_labels, duplicates, errors, total.
    """
    root = Path(dataset_root)
    result: Dict[str, Any] = {
        "valid":          False,
        "genuine_count":  0,
        "tampered_count": 0,
        "missing_files":  [],
        "invalid_labels": [],
        "duplicates":     [],
        "errors":         [],
        "total":          0,
    }

    if not root.exists():
        result["errors"].append(f"Dataset root does not exist: {root}")
        return result

    manifest = root / "manifest.csv"
    if not manifest.exists():
        result["errors"].append(f"manifest.csv not found at {manifest}")
        return result

    seen_paths: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    try:
        with open(manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):
                rows.append(row)
                img_rel  = row.get("image_path", "").strip()
                label    = row.get("label", "").strip()
                base_id  = row.get("base_document_id", "").strip()

                img_path = root / img_rel if img_rel else None

                if label not in ("0", "1"):
                    result["invalid_labels"].append(f"Row {i}: '{label}'")
                else:
                    if int(label) == LABEL_GENUINE:
                        result["genuine_count"] += 1
                    else:
                        result["tampered_count"] += 1

                if img_path and not img_path.exists():
                    result["missing_files"].append(str(img_path))

                if img_rel in seen_paths:
                    result["duplicates"].append(img_rel)
                else:
                    seen_paths[img_rel] = i

    except Exception as exc:
        result["errors"].append(f"Manifest read error: {exc}")
        return result

    result["total"] = len(rows)

    result["valid"] = (
        not result["missing_files"]
        and not result["invalid_labels"]
        and not result["duplicates"]
        and not result["errors"]
        and result["total"] > 0
    )
    return result


# ── Main generation orchestrator ──────────────────────────────────────────────

def build_dataset(
    source_dir:   Union[str, Path],
    dataset_root: Union[str, Path],
    seed:         int = 42,
    photo_region: Optional[Tuple[int,int,int,int]] = None,
) -> Dict[str, Any]:
    """
    Orchestrate full Phase 2 dataset generation.

    Steps:
      1. Discover source images.
      2. For each source image: save a genuine sample + 4 tampered variants
         (using the next image in the list as the donor).
      3. Write full and Phase-3-compatible manifests.
      4. Write JSON summary.

    Args:
        source_dir:   Directory containing original passport images.
        dataset_root: Output directory for the dataset.
        seed:         Global RNG seed.
        photo_region: Portrait box override (default: synthetic layout).

    Returns:
        Dict with: success, records, summary_path, full_manifest_path,
                   p3_manifest_path, errors.
    """
    source_dir   = Path(source_dir)
    dataset_root = Path(dataset_root)

    result: Dict[str, Any] = {
        "success":             False,
        "records":             [],
        "summary_path":        None,
        "full_manifest_path":  None,
        "p3_manifest_path":    None,
        "errors":              [],
    }

    try:
        sources = discover_source_images(source_dir)
    except FileNotFoundError as exc:
        result["errors"].append(str(exc))
        return result

    if len(sources) < 2:
        result["errors"].append(
            f"Need at least 2 source images for cross-image operations, "
            f"found {len(sources)} in {source_dir}."
        )
        return result

    all_records: List[Dict[str, Any]] = []

    for i, src in enumerate(sources):
        base_doc_id = src.stem   # e.g. "passport1"
        donor       = sources[(i + 1) % len(sources)]   # next image, wraps around

        # Genuine
        genuine_rec = create_genuine_sample(src, dataset_root, base_doc_id)
        all_records.append(genuine_rec)

        # Tampered variants
        tampered_recs = create_tampered_samples(
            source_path=src,
            donor_path=donor,
            output_dir=dataset_root,
            base_doc_id=base_doc_id,
            seed=seed,
            photo_region=photo_region,
        )
        all_records.extend(tampered_recs)

    result["records"] = all_records

    # Write manifests
    try:
        full_path, p3_path = write_manifests(all_records, dataset_root)
        result["full_manifest_path"] = str(full_path)
        result["p3_manifest_path"]   = str(p3_path)
    except Exception as exc:
        result["errors"].append(f"Manifest write failed: {exc}")
        return result

    # Write summary
    try:
        summary_path = write_summary(all_records, dataset_root, str(source_dir))
        result["summary_path"] = str(summary_path)
    except Exception as exc:
        result["errors"].append(f"Summary write failed: {exc}")

    result["success"] = True
    return result
