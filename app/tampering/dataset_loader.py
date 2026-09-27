"""
Dataset Loader — Module 3, Phase 3.

Expected folder layout (created by Phase 2):

    <dataset_root>/
        genuine/          <- class 0
            <base_doc_id>/
                image1.jpg
                ...
        tampered/         <- class 1
            <base_doc_id>/
                image1.jpg
                ...
        manifest.csv      <- optional; columns: image_path, label, base_document_id

A manifest CSV is preferred when it exists because it preserves the
base_document_id needed for grouped splits. If only folders exist a
manifest is derived automatically (subfolder name → base_document_id).

Labels: 0 = GENUINE, 1 = TAMPERED
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_GENUINE  = 0
LABEL_TAMPERED = 1
CLASS_NAMES    = {LABEL_GENUINE: "GENUINE", LABEL_TAMPERED: "TAMPERED"}
_IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Dataset record type ───────────────────────────────────────────────────────
# Each sample: {"image_path": str, "label": int, "base_document_id": str}
Sample = Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_dataset_readiness(dataset_root: str | Path) -> Dict[str, Any]:
    """
    Inspect whether a Phase 2 dataset exists and is minimally usable.

    Returns a dict with:
        ready           (bool)
        genuine_count   (int)
        tampered_count  (int)
        has_manifest    (bool)
        issues          (list[str])
        dataset_root    (str)
    """
    root = Path(dataset_root)
    result: Dict[str, Any] = {
        "ready":          False,
        "genuine_count":  0,
        "tampered_count": 0,
        "has_manifest":   False,
        "issues":         [],
        "dataset_root":   str(root),
    }

    if not root.exists():
        result["issues"].append(
            f"Dataset root does not exist: {root}\n"
            "  → Phase 2 (synthetic dataset generation) must be completed first."
        )
        return result

    manifest = root / "manifest.csv"
    result["has_manifest"] = manifest.exists()

    genuine_dir  = root / "genuine"
    tampered_dir = root / "tampered"

    # Count images (from manifest or folders)
    if manifest.exists():
        try:
            samples = _load_manifest(manifest, root)
            result["genuine_count"]  = sum(1 for s in samples if s["label"] == LABEL_GENUINE)
            result["tampered_count"] = sum(1 for s in samples if s["label"] == LABEL_TAMPERED)
        except Exception as exc:
            result["issues"].append(f"Manifest parse error: {exc}")
    else:
        g = _count_images_in_dir(genuine_dir)
        t = _count_images_in_dir(tampered_dir)
        result["genuine_count"]  = g
        result["tampered_count"] = t
        if not genuine_dir.exists():
            result["issues"].append(f"Missing 'genuine/' folder under {root}")
        if not tampered_dir.exists():
            result["issues"].append(f"Missing 'tampered/' folder under {root}")

    if result["genuine_count"] == 0:
        result["issues"].append("No genuine images found.")
    if result["tampered_count"] == 0:
        result["issues"].append("No tampered images found.")

    total = result["genuine_count"] + result["tampered_count"]
    if total < 6:
        result["issues"].append(
            f"Only {total} total images — need at least 6 for a 3-way grouped split."
        )

    result["ready"] = len(result["issues"]) == 0
    return result


def _count_images_in_dir(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _load_manifest(manifest: Path, root: Path) -> List[Sample]:
    """Parse manifest.csv into a list of Sample dicts. Raises on invalid rows."""
    samples: List[Sample] = []
    errors: List[str] = []

    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label", "base_document_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"manifest.csv must have columns: {required}. "
                f"Found: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):  # row 1 = header
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = root / img_path
            label_raw = row["label"].strip()
            base_id   = row["base_document_id"].strip()

            if label_raw not in ("0", "1"):
                errors.append(f"Row {i}: invalid label '{label_raw}' (must be 0 or 1)")
                continue
            if not img_path.exists():
                errors.append(f"Row {i}: image not found: {img_path}")
                continue
            if img_path.suffix.lower() not in _IMG_EXTS:
                errors.append(f"Row {i}: unsupported format '{img_path.suffix}'")
                continue
            if not base_id:
                errors.append(f"Row {i}: empty base_document_id")
                continue

            samples.append({
                "image_path":       str(img_path),
                "label":            int(label_raw),
                "base_document_id": base_id,
                "affected_region":  row.get("affected_region", "").strip() if "affected_region" in row else "",
            })

    if errors:
        raise ValueError(
            f"{len(errors)} invalid row(s) in manifest.csv:\n" + "\n".join(errors)
        )
    return samples


def _load_from_folders(root: Path) -> List[Sample]:
    """Derive samples from genuine/ and tampered/ sub-folders.
    Sub-folder name under each class = base_document_id."""
    samples: List[Sample] = []
    errors: List[str] = []

    for label, subdir_name in ((LABEL_GENUINE, "genuine"), (LABEL_TAMPERED, "tampered")):
        class_dir = root / subdir_name
        if not class_dir.exists():
            continue
        for doc_dir in sorted(class_dir.iterdir()):
            if not doc_dir.is_dir():
                # flat structure — use class dir name as base_id
                p = doc_dir
                if p.suffix.lower() in _IMG_EXTS:
                    samples.append({
                        "image_path":       str(p),
                        "label":            label,
                        "base_document_id": subdir_name,
                    })
                continue
            base_id = doc_dir.name
            for p in sorted(doc_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                    samples.append({
                        "image_path":       str(p),
                        "label":            label,
                        "base_document_id": base_id,
                    })

    if errors:
        raise ValueError("\n".join(errors))
    return samples


# ── Grouped split ─────────────────────────────────────────────────────────────

def grouped_split(
    samples: List[Sample],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Split samples into train/val/test ensuring no base_document_id
    appears in more than one split.

    Raises ValueError if the dataset is too small for a valid split.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    # Group by base_document_id
    by_doc: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_doc[s["base_document_id"]].append(s)

    doc_ids = sorted(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    if n < 3:
        raise ValueError(
            f"Only {n} unique base_document_id(s). Need at least 3 for a grouped split. "
            "Generate more synthetic data (Phase 2) before training."
        )

    n_val  = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Dataset has {n} documents — too few for a 3-way grouped split "
            f"({n_train} train docs). Add more documents via Phase 2."
        )

    train_ids = doc_ids[:n_train]
    val_ids   = doc_ids[n_train:n_train + n_val]
    test_ids  = doc_ids[n_train + n_val:]

    def gather(ids: List[str]) -> List[Sample]:
        result = []
        for d in ids:
            result.extend(by_doc[d])
        return result

    return gather(train_ids), gather(val_ids), gather(test_ids)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_dataset(
    dataset_root: str | Path,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
) -> Dict[str, Any]:
    """
    Load the Phase 2 dataset and return grouped train/val/test splits.

    Returns dict with:
        success       (bool)
        train         (list[Sample])
        val           (list[Sample])
        test          (list[Sample])
        class_counts  (dict)
        split_counts  (dict)
        source        ("manifest" | "folders")
        errors        (list[str])
    """
    result: Dict[str, Any] = {
        "success":      False,
        "train":        [],
        "val":          [],
        "test":         [],
        "class_counts": {},
        "split_counts": {},
        "source":       None,
        "errors":       [],
    }

    root = Path(dataset_root)
    manifest = root / "manifest.csv"

    try:
        if manifest.exists():
            samples = _load_manifest(manifest, root)
            result["source"] = "manifest"
        else:
            samples = _load_from_folders(root)
            result["source"] = "folders"
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    if not samples:
        result["errors"].append("Dataset is empty.")
        return result

    # Class counts
    class_counts: Dict[str, int] = {CLASS_NAMES[LABEL_GENUINE]: 0, CLASS_NAMES[LABEL_TAMPERED]: 0}
    for s in samples:
        class_counts[CLASS_NAMES[s["label"]]] += 1
    result["class_counts"] = class_counts

    # Grouped split
    try:
        train, val, test = grouped_split(samples, train_ratio, val_ratio, seed)
    except ValueError as exc:
        result["errors"].append(str(exc))
        return result

    result["train"] = train
    result["val"]   = val
    result["test"]  = test
    result["split_counts"] = {
        "train": len(train),
        "val":   len(val),
        "test":  len(test),
    }
    result["success"] = True
    return result
