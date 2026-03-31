"""Language model training loop."""

import argparse
import contextlib
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


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
        "args": vars(args),
    }

    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    device: torch.device,
) -> int:
    """Load training checkpoint from file.

    Returns:
        The training step to resume from.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

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


def create_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Create a cosine LR scheduler with linear warmup.

    During warmup: LR scale increases linearly from 0 to 1 over warmup_steps steps.
    After warmup: LR scale decays following a cosine curve down to min_lr_ratio.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        warmup_steps: Number of steps for linear warmup.
        max_steps: Total number of training steps.
        min_lr_ratio: Final LR scale as a fraction of the base LR. Defaults to 0.1.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_cosine_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_cosine_lambda)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Compute average validation loss over a number of batches.

    Args:
        model: The model to evaluate.
        val_loader: Validation DataLoader.
        device: Device to run evaluation on.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Average loss(cross-entropy) over the evaluated batches.
    """
    model.eval()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    )

    total_loss = 0.0
    n_batches = 0

    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)
