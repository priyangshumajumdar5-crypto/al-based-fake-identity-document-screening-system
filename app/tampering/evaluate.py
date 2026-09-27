"""
Evaluation Module — Module 3, Phase 3.

Evaluates a trained EfficientNet-B0 checkpoint on the held-out test split.
Uses scikit-learn for metrics (must be installed — see dependency note).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from app.tampering.model import (
    build_model,
    get_device,
    load_checkpoint,
    NUM_CLASSES,
    CLASS_NAMES,
)
from app.tampering.dataset_loader import Sample
from app.tampering.inference import predict


def evaluate(
    test_samples:     List[Sample],
    checkpoint_path:  str | Path,
    output_dir:       Optional[str | Path] = None,
    batch_size:       int = 16,
) -> Dict[str, Any]:
    """
    Evaluate a trained EfficientNet-B0 checkpoint on the held-out test split
    using full patch-based inference.

    Args:
        test_samples:    Test split (list of Sample dicts from dataset_loader).
        checkpoint_path: Path to the .pt checkpoint saved during training.
        output_dir:      If provided, saves results as JSON + CSV there.
        batch_size:      Batch size for evaluation.

    Returns:
        Dict with: accuracy, precision, recall, f1, roc_auc, confusion_matrix,
                   classification_report, per_sample_preds, errors.
    """
    result: Dict[str, Any] = {
        "accuracy":               None,
        "precision":              None,
        "recall":                 None,
        "f1":                     None,
        "roc_auc":                None,
        "confusion_matrix":       None,
        "classification_report":  None,
        "per_sample_preds":       [],
        "errors":                 [],
    }

    if not test_samples:
        result["errors"].append("test_samples is empty — nothing to evaluate.")
        return result

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        result["errors"].append(f"Checkpoint not found: {checkpoint_path}")
        return result

    device = get_device()
    all_labels: List[int] = []
    all_preds:  List[int] = []
    all_scores: List[float] = []

    for s in test_samples:
        img_path = s["image_path"]
        true_lbl = s["label"]
        pred_res = predict(image_path=img_path, checkpoint_path=checkpoint_path, device=device)

        if not pred_res.get("success"):
            result["errors"].append(f"Inference failed for {img_path}: {pred_res.get('errors')}")
            pred_score = 0.5
            pred_lbl = 1 if pred_score >= 0.5 else 0
        else:
            pred_score = pred_res.get("score_tampered", 0.0)
            pred_lbl = pred_res.get("predicted_class", 1 if pred_score >= 0.5 else 0)

        all_labels.append(true_lbl)
        all_preds.append(pred_lbl)
        all_scores.append(pred_score)

        result["per_sample_preds"].append({
            "image_path":     img_path,
            "base_doc_id":    s.get("base_document_id", ""),
            "true_label":     true_lbl,
            "pred_label":     pred_lbl,
            "score_tampered": pred_score,
            "correct":        true_lbl == pred_lbl,
        })

    # ── scikit-learn metrics ──────────────────────────────────────────────────
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            classification_report,
        )

        class_labels = list(range(NUM_CLASSES))
        target_names = [CLASS_NAMES[c] for c in class_labels]

        result["accuracy"]  = round(float(accuracy_score(all_labels, all_preds)), 5)
        result["precision"] = round(float(precision_score(
            all_labels, all_preds, average="weighted",
            labels=class_labels, zero_division=0
        )), 5)
        result["recall"] = round(float(recall_score(
            all_labels, all_preds, average="weighted",
            labels=class_labels, zero_division=0
        )), 5)
        result["f1"] = round(float(f1_score(
            all_labels, all_preds, average="weighted",
            labels=class_labels, zero_division=0
        )), 5)

        # ROC-AUC calculation
        try:
            if len(set(all_labels)) > 1:
                result["roc_auc"] = round(float(roc_auc_score(all_labels, all_scores)), 5)
            else:
                result["roc_auc"] = None
                result["errors"].append("ROC-AUC not computed: test split contains only 1 class.")
        except Exception as exc:
            result["roc_auc"] = None
            result["errors"].append(f"ROC-AUC computation failed: {exc}")

        cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
        result["confusion_matrix"] = cm.tolist()
        result["classification_report"] = classification_report(
            all_labels, all_preds,
            labels=class_labels,
            target_names=target_names,
            zero_division=0,
        )

    except ImportError:
        # Fallback: manual accuracy only
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        result["accuracy"] = round(correct / len(all_labels), 5)
        result["errors"].append(
            "scikit-learn not installed — only accuracy computed. "
            "Install with: pip install scikit-learn"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON (full report)
        json_path = out / "test_evaluation.json"
        exportable = {k: v for k, v in result.items() if k != "per_sample_preds"}
        with open(json_path, "w") as f:
            json.dump(exportable, f, indent=2)

        # CSV (per-sample)
        csv_path = out / "per_sample_predictions.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "base_doc_id", "true_label", "pred_label", "score_tampered", "correct"],
                extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(result["per_sample_preds"])

        result["output_json"] = str(json_path)
        result["output_csv"]  = str(csv_path)
        print(f"Results saved → {json_path}, {csv_path}")

    return result
