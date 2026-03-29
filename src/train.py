"""Language model training loop."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler


def set_seed(seed: int) -> None:
    """Manually set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    step: int,
    args: argparse.Namespace,
) -> None:
    """Save training checkpoint to file."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }

    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    device: torch.device,
) -> int:
    """Load training checkpoint from file.

    Returns the training step to continue from.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return int(ckpt.get("step", 0))


def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into weight-decay and no-weight-decay groups.

    Applies weight decay only to "true weights" (Linear weights).
    Biases, normalization, and embedding parameters are excluded.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if "norm" in name or "ln" in name or "embed" in name or "embedding" in name or "bias" in name:
            no_decay.append(p)
            continue

        decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
