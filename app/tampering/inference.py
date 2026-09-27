"""
Inference Module — Module 3 (patch-based).
Scores are raw softmax outputs, NOT calibrated fraud probabilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import UnidentifiedImageError

from app.tampering import patches as P
from app.tampering.model import (
    build_model,
    get_device,
    load_checkpoint,
    NUM_CLASSES,
)

_MODEL_CACHE: Dict[str, Any] = {}


def _get_model(checkpoint_path: Path, device: torch.device):
    """Load model once and reuse it (reloads if the checkpoint file changes)."""
    key = f"{checkpoint_path.resolve()}|{checkpoint_path.stat().st_mtime}|{device}"
    if key not in _MODEL_CACHE:
        ckpt = load_checkpoint(checkpoint_path, device=device)
        cfg = ckpt.get("config", {})
        model = build_model(
            pretrained=False,
            freeze_backbone=False,
            num_classes=cfg.get("num_classes", NUM_CLASSES),
        )
        state_dict = ckpt.get("model_state", {})
        # Strip DataParallel 'module.' prefix if present
        cleaned_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError as err:
            import logging
            logging.warning("Strict state_dict load failed (%s); falling back with loud warning.", err)
            model.load_state_dict(cleaned_state_dict, strict=False)
        model.to(device)
        model.eval()
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = (model, ckpt)
    return _MODEL_CACHE[key]


def predict(
    image_path:      Union[str, Path],
    checkpoint_path: Union[str, Path],
    device:          Optional[torch.device] = None,
    top_k:           int = 3,
) -> Dict[str, Any]:
    """
    Returns dict with: success, band ("GENUINE"|"UNCERTAIN"|"TAMPERED"|"MODEL_UNAVAILABLE"),
    predicted_label (= band), predicted_class, score_genuine, score_tampered,
    suspicious_regions (boxes in the 768-px-wide normalised image), n_patches,
    checkpoint_info, model_config, warnings, errors.
    """
    result: Dict[str, Any] = {
        "success":            False,
        "band":               "MODEL_UNAVAILABLE",
        "predicted_class":    None,
        "predicted_label":    "MODEL_UNAVAILABLE",
        "score_genuine":      None,
        "score_tampered":     None,
        "suspicious_regions": [],
        "n_patches":          0,
        "checkpoint_info":    {},
        "model_config":       {},
        "warnings":           [],
        "errors":             [],
    }

    device          = device or get_device()
    image_path      = Path(image_path)
    checkpoint_path = Path(checkpoint_path)

    if not image_path.exists():
        result["errors"].append(f"Image not found: {image_path}")
        return result
    if not checkpoint_path.exists():
        result["errors"].append(f"Checkpoint not found: {checkpoint_path}")
        result["warnings"].append("EfficientNet-B0 checkpoint file missing. Model signal is unavailable.")
        return result

    try:
        img = P.load_normalised(image_path)
    except UnidentifiedImageError:
        result["errors"].append(f"Cannot identify image: {image_path}")
        return result
    except Exception as exc:
        result["errors"].append(f"Image load failed: {exc}")
        return result

    try:
        model, ckpt = _get_model(checkpoint_path, device)
    except Exception as exc:
        result["errors"].append(f"Checkpoint/model load failed: {exc}")
        return result

    cfg = ckpt.get("config", {})
    result["checkpoint_info"] = {"epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc")}
    result["model_config"] = cfg
    if not cfg.get("patch_based"):
        result["warnings"].append(
            "This checkpoint was trained with the OLD whole-image pipeline. "
            "Retrain with the patch-based train() for meaningful results."
        )

    try:
        arr, _ = P.pad_to_min(np.asarray(img))
        h, w = arr.shape[:2]
        boxes = P.tile_boxes(h, w)
        batch = torch.stack(
            [P.patch_to_tensor(arr[y0:y1, x0:x1]) for (x0, y0, x1, y1) in boxes]
        ).to(device)
        scores = []
        with torch.no_grad():
            for i in range(0, len(batch), 32):
                logits = model(batch[i:i + 32])
                scores.append(F.softmax(logits, dim=1)[:, 1].cpu())
        p_t = torch.cat(scores).numpy()
    except Exception as exc:
        result["errors"].append(f"Inference failed: {exc}")
        return result

    order = np.argsort(-p_t)
    k = max(1, min(top_k, len(p_t)))
    score_tampered = float(p_t[order[:k]].mean())
    score_genuine  = float(1.0 - score_tampered)
    difference     = float(score_tampered - score_genuine)

    # Multi-tier classification logic based on score difference (Tampered Score - Genuine Score).
    # KEY RULE: difference = 0.00 means exact 50-50. Any positive value means tampered side leads.
    if difference <= -0.25:
        classification_result   = "LIKELY_GENUINE"
        classification_category = "STRONG_GENUINE_LEAN"
        classification_label    = "Strongly Genuine (High Confidence)"
        classification_desc     = (
            f"Genuine score heavily dominates tampered score (Difference: {difference:+.3f}). "
            "High confidence that no visual tampering patterns exist across scanned patches. Document appears authentic."
        )
        predicted_class = 0
    elif -0.25 < difference <= -0.10:
        classification_result   = "LIKELY_GENUINE"
        classification_category = "GENUINE_LEAN"
        classification_label    = "Genuine-Leaning (Moderate Confidence)"
        classification_desc     = (
            f"Genuine score exceeds tampered score (Difference: {difference:+.3f}). "
            "Low overall tampering risk detected across scanned patches. Document appears genuine."
        )
        predicted_class = 0
    elif -0.10 < difference < 0.00:
        # Negative but small — tampering cannot be ruled out
        classification_result   = "UNCERTAIN"
        classification_category = "WEAK_GENUINE_LEAN"
        classification_label    = "Borderline Genuine (Very Small Gap — Review Advised)"
        classification_desc     = (
            f"Genuine score leads by only a very small margin (Difference: {difference:+.3f}). "
            "Gap is too narrow to confidently clear the document. Tampering patterns cannot be excluded. "
            "Secondary officer review is required."
        )
        predicted_class = 0
    elif difference == 0.00:
        # Exact zero — mathematically 50-50
        classification_result   = "INCONCLUSIVE"
        classification_category = "INCONCLUSIVE"
        classification_label    = "Inconclusive (Exact 50–50 Probability)"
        classification_desc     = (
            f"Score difference is exactly zero ({difference:+.3f}), meaning the model assigns equal probability "
            "to genuine and tampered classes. Mandatory manual verification required."
        )
        predicted_class = 0
    elif 0.00 < difference <= 0.10:
        # Positive — tampered side leads, even if small
        classification_result   = "POTENTIALLY_TAMPERED"
        classification_category = "WEAK_TAMPERING_LEAN"
        classification_label    = "Borderline Tampering (Tampered Side Leads — Verify)"
        classification_desc     = (
            f"Tampered score exceeds genuine score (Difference: {difference:+.3f}). "
            "The model assigns higher probability to the tampered class. "
            "Localized patch anomalies detected. Document flagged — secondary officer verification mandatory."
        )
        predicted_class = 1
    elif 0.10 < difference <= 0.25:
        classification_result   = "POTENTIALLY_TAMPERED"
        classification_category = "TAMPERING_LEAN"
        classification_label    = "Tampering-Leaning (Moderate Risk)"
        classification_desc     = (
            f"Tampered score noticeably exceeds genuine score (Difference: {difference:+.3f}). "
            "Clear suspicious patterns identified across document patches. Secondary verification required."
        )
        predicted_class = 1
    else:  # difference > 0.25
        classification_result   = "POTENTIALLY_TAMPERED"
        classification_category = "STRONG_TAMPERING_LEAN"
        classification_label    = "Strongly Tampered (High Risk)"
        classification_desc     = (
            f"Tampered score heavily dominates genuine score (Difference: {difference:+.3f}). "
            "High confidence of visual tampering or image manipulation across scanned patches. Flagged for fraud investigation."
        )
        predicted_class = 1

    result["suspicious_regions"] = [
        {"box": list(boxes[j]), "score": round(float(p_t[j]), 4)}
        for j in order[:k] if p_t[j] >= 0.5
    ]
    result["n_patches"]               = len(boxes)
    result["band"]                    = classification_result
    result["predicted_label"]         = classification_label
    result["predicted_class"]         = predicted_class
    result["classification_result"]   = classification_result
    result["classification_category"] = classification_category
    result["classification_label"]    = classification_label
    result["classification_desc"]     = classification_desc
    result["score_tampered"]          = round(score_tampered, 5)
    result["score_genuine"]           = round(score_genuine, 5)
    result["score_difference"]        = round(difference, 5)
    result["success"]                 = True
    result["warnings"].append(classification_desc)
    return result