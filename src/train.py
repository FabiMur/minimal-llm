"""Language model training loop."""

import argparse
from pathlib import Path

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
    # Ensure directory exists
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint payload
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }

    # Save to file
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
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load checkpoint states into model, optimizer, scheduler and scaler
    model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return int(ckpt.get("step", 0))
