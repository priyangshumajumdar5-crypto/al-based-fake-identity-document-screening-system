"""
Training Pipeline — Module 3, Phase 3.

Trains the EfficientNet-B0 classifier on the Phase 2 dataset.
Do NOT run this file directly; import and call train() from the notebook.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import random
import numpy as np
from app.tampering import patches as P

from app.tampering.model import (
    build_model,
    get_device,
    get_transforms,
    save_checkpoint,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
)
from app.tampering.dataset_loader import Sample, LABEL_GENUINE, LABEL_TAMPERED


# ── Internal PyTorch Dataset ──────────────────────────────────────────────────

class _PassportDataset(Dataset):
    def __init__(self, samples: List[Sample], split: str = "val"):
        self.samples   = samples
        self.transform = get_transforms(split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        tensor = self.transform(img)
        label  = torch.tensor(s["label"], dtype=torch.long)
        return tensor, label

# ── Patch dataset (replaces whole-image training) ─────────────────────────────

class _PatchDataset(Dataset):
    """Random native-resolution 224x224 patches. label 1 = patch contains edited pixels."""

    def __init__(self, samples: List[Sample], split: str = "train",
                 patches_per_image: int = 20, seed: int = 42):
        self.samples = samples
        self.train   = (split == "train")
        self.ppi     = patches_per_image
        self.seed    = seed
        self._cache: Dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.samples) * self.ppi

    def _load(self, i: int):
        if i not in self._cache:
            s = self.samples[i]
            img = P.load_normalised(s["image_path"])
            mask = None
            if s["label"] == LABEL_TAMPERED:
                ref = P.genuine_reference_path(s["image_path"])
                if ref is not None:
                    mask = P.tamper_mask(img, ref)
                elif s.get("affected_region"):
                    try:
                        import ast
                        reg_raw = s["affected_region"]
                        reg = ast.literal_eval(reg_raw) if isinstance(reg_raw, str) else reg_raw
                        if isinstance(reg, (tuple, list)) and len(reg) == 4:
                            mask = P.mask_from_affected_region(img.height, img.width, reg)
                    except Exception:
                        mask = None
            self._cache[i] = (np.asarray(img), mask)
        return self._cache[i]

    def __getitem__(self, k: int):
        arr, mask = self._load(k // self.ppi)
        rng = random.Random() if self.train else random.Random(self.seed * 100003 + k)
        patch, label = P.sample_patch(arr, mask, rng.random() < 0.5, rng)
        patch = P.degrade(patch, rng)
        return P.patch_to_tensor(patch), torch.tensor(label, dtype=torch.long)

def _make_weighted_sampler(samples: List[Sample]) -> WeightedRandomSampler:
    """Return a sampler that up-weights the minority class."""
    labels = [s["label"] for s in samples]
    counts = [labels.count(c) for c in range(NUM_CLASSES)]
    # Avoid zero-division if a class is absent
    weights_per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Training function ─────────────────────────────────────────────────────────

def train(
    train_samples:   List[Sample],
    val_samples:     List[Sample],
    checkpoint_dir:  str | Path,
    pretrained:      bool  = True,
    freeze_backbone: bool  = True,
    unfreeze_after:  Optional[int] = 5,   # epoch to unfreeze backbone (None = keep frozen)
    lr:              float = 1e-4,
    batch_size:      int   = 16,
    epochs:          int   = 20,
    patience:        int   = 5,
    weight_decay:    float = 1e-4,
    seed:            int   = 42,
    use_weighted_sampler: bool = True,
) -> Dict[str, Any]:
    """
    Train the EfficientNet-B0 classifier.

    Args:
        train_samples:   Training split (list of Sample dicts).
        val_samples:     Validation split.
        checkpoint_dir:  Directory to save best checkpoint.
        pretrained:      Use ImageNet pretrained weights.
        freeze_backbone: Start with backbone frozen.
        unfreeze_after:  Epoch (1-indexed) after which to unfreeze backbone. None = never.
        lr:              Initial learning rate.
        batch_size:      DataLoader batch size.
        epochs:          Maximum training epochs.
        patience:        Early-stopping patience (epochs without val improvement).
        weight_decay:    AdamW weight decay.
        seed:            Random seed.
        use_weighted_sampler: Up-weight minority class in train loader.

    Returns:
        Dict with: history, best_val_acc, best_epoch, checkpoint_path, config, errors.
    """
    torch.manual_seed(seed)
    device = get_device()

    result: Dict[str, Any] = {
        "history":          [],
        "best_val_acc":     0.0,
        "best_epoch":       0,
        "checkpoint_path":  None,
        "config":           {},
        "errors":           [],
    }

    config = {
        "pretrained":           pretrained,
        "freeze_backbone":      freeze_backbone,
        "unfreeze_after":       unfreeze_after,
        "lr":                   lr,
        "batch_size":           batch_size,
        "epochs":               epochs,
        "patience":             patience,
        "weight_decay":         weight_decay,
        "seed":                 seed,
        "use_weighted_sampler": use_weighted_sampler,
        "input_size":           list(INPUT_SIZE),
        "patch_based":          True,
        "imagenet_mean":        IMAGENET_MEAN,
        "imagenet_std":         IMAGENET_STD,
        "num_classes":          NUM_CLASSES,
        "device":               str(device),
        "train_samples":        len(train_samples),
        "val_samples":          len(val_samples),
    }
    result["config"] = config

    if not train_samples:
        result["errors"].append("train_samples is empty.")
        return result
    if not val_samples:
        result["errors"].append("val_samples is empty.")
        return result

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = _PatchDataset(train_samples, split="train", patches_per_image=12)
    val_ds   = _PatchDataset(val_samples,   split="val",   patches_per_image=20)

    sampler = _make_weighted_sampler(train_samples) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────
    model = build_model(pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    ckpt_path    = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Optionally unfreeze backbone
        if unfreeze_after is not None and epoch == unfreeze_after:
            from app.tampering.model import unfreeze_backbone
            model = unfreeze_backbone(model)
            # Re-create optimizer to include all params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr * 0.1, weight_decay=weight_decay
            )

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc      = train_correct / max(train_total, 1)

        # ── Val epoch ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc      = val_correct / max(val_total, 1)

        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_acc":  round(train_acc, 5),
            "val_loss":   round(avg_val_loss, 5),
            "val_acc":    round(val_acc, 5),
            "elapsed_s":  round(elapsed, 2),
        }
        result["history"].append(row)
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # ── Checkpoint best model ─────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            ckpt_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                config=config,
                save_dir=checkpoint_dir,
                filename="best_checkpoint.pt",
            )
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    result["best_val_acc"]    = best_val_acc
    result["best_epoch"]      = best_epoch
    result["checkpoint_path"] = ckpt_path
    return result
