# src/train.py
import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from data import create_bin_dataloaders, load_meta
from model import GPTConfig, GPTModel


def set_seed(seed: int) -> None:
    """
    Manually set random seed for reproducibility.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get available device: CUDA > MPS > CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_dumps_safe(obj) -> str:
    """
    Transform Python object to JSON string, handling Path objects.
    """

    # If Path objects are present, convert them to strings
    def default(o):
        if isinstance(o, Path):
            return str(o)
        return repr(o)

    return json.dumps(obj, indent=2, default=default)



def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    step: int,
    args: argparse.Namespace,
) -> None:
    """
    Save training checkpoint to disk.
    """

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

    # Save to disk
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> int:
    """
    Load training checkpoint from disk.
    Returns the training step to resume from.
    """

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load states into model, optimizer, scheduler and scaler
    model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
        
    return int(ckpt.get("step", 0))


@torch.no_grad() # Disable gradient calculation for evaluation
def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> float:
    """
    Evaluate model on validation set and return average loss.
    """

    model.eval() # Set model to evaluation mode
    losses = []
    it = iter(val_loader)

    for _ in range(max_batches):
        try:
            x, y = next(it) # Get next batch (x: inputs, y: targets)
        except StopIteration:
            break

        # Move data to device (e.g., GPU or CPU)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass with automatic mixed precision if enabled
        with autocast(device_type=device.type, enabled=amp):
            loss = model(x, y) # Assume model returns loss directly

        # Collect loss
        losses.append(float(loss.item()))

    model.train() # Set model back to training mode

    return float(sum(losses) / len(losses)) if losses else float("nan")



def make_cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """
    Create a cosine learning rate scheduler with linear warmup and cosine decay.
    """

    def lr_lambda(step: int) -> float:

        # Linear warmup
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0) # Clamp to [0, 1]

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)