"""
EfficientNet-B0 Model — Module 3, Phase 3.

Transfer-learning wrapper around torchvision's EfficientNet-B0.
Final classifier head is replaced with a 2-class output (GENUINE / TAMPERED).

Phase 5 will connect this to the Streamlit dashboard and FastAPI.
Do NOT use this file for EfficientNet-based preprocessing of ELA images
unless the Phase 2 dataset is explicitly built that way.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ── Pretrained weights ────────────────────────────────────────────────────────
# torchvision >= 0.13 exposes EfficientNet_B0_Weights.
# We import with a fallback so the module loads even if torchvision is old.
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _WEIGHTS_AVAILABLE = True
except ImportError:
    from torchvision.models import efficientnet_b0
    EfficientNet_B0_Weights = None          # type: ignore[assignment]
    _WEIGHTS_AVAILABLE = False

NUM_CLASSES = 2
CLASS_NAMES = {0: "GENUINE", 1: "TAMPERED"}

# Input size required by EfficientNet-B0 ImageNet weights
INPUT_SIZE: Tuple[int, int] = (224, 224)

# ImageNet normalisation stats used by EfficientNet_B0_Weights.IMAGENET1K_V1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def get_transforms(split: str = "val"):
    """
    Return torchvision transforms for a given split.

    Args:
        split: "train", "val", or "test" / "inference".
               Training split gets mild augmentation; others get only
               resize + centre-crop + normalisation.

    Returns:
        torchvision.transforms.Compose object.
    """
    from torchvision import transforms

    base = [
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train":
        augment = [
            transforms.Resize(256),
            transforms.RandomCrop(INPUT_SIZE[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        from torchvision.transforms import Compose
        return Compose(augment)

    from torchvision.transforms import Compose
    return Compose(base)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    pretrained:      bool = True,
    freeze_backbone: bool = True,
    num_classes:     int  = NUM_CLASSES,
) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.

    Args:
        pretrained:      Load ImageNet pretrained weights if available.
        freeze_backbone: Freeze all layers except the final head.
        num_classes:     Output classes (default 2).

    Returns:
        torch.nn.Module ready for training or inference.
    """
    if pretrained and _WEIGHTS_AVAILABLE:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    elif pretrained and not _WEIGHTS_AVAILABLE:
        # Older torchvision API
        model = efficientnet_b0(pretrained=True)          # type: ignore[call-arg]
    else:
        model = efficientnet_b0(weights=None) if _WEIGHTS_AVAILABLE else efficientnet_b0(pretrained=False)  # type: ignore[call-arg]

    # Replace classifier head: EfficientNet_B0 last layer is model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    val_acc:    float,
    config:     Dict[str, Any],
    save_dir:   str | Path,
    filename:   str = "best_checkpoint.pt",
) -> str:
    """Save model state_dict + config to save_dir/filename."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / filename
    torch.save(
        {
            "epoch":       epoch,
            "val_acc":     val_acc,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config":      config,
        },
        ckpt_path,
    )
    # Also persist config as JSON for inspection without PyTorch
    cfg_path = save_dir / "training_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    return str(ckpt_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    device:          Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint dict from disk.

    Returns the raw dict (model_state, config, epoch, val_acc, optim_state).
    Does NOT rebuild the model — use build_model() then load_state_dict().
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt
