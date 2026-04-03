"""Language model training loop."""

import argparse
import contextlib
import math
import time
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from minimal_llm.data.data_loaders import create_bin_dataloaders
from minimal_llm.model import ModelConfig, TransformerLM


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a minimal transformer language model.")

    # IO
    parser.add_argument("--run_name", type=str, default=str(date.today().isoformat()), help="Training run name")
    parser.add_argument("--meta", type=Path, default=Path("artifacts/meta.json"), help="Path to meta.json.")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/checkpoints"), help="Dir to save checkpoints.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation and logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log loss every N steps.")
    parser.add_argument("--eval_interval", type=int, default=500, help="Run validation every N steps.")
    parser.add_argument("--eval_batches", type=int, default=50, help="Max batches per validation run.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps.")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=16)

    return parser.parse_args()


def main() -> None:
    """Run training loop."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Device: {device}")

    # Model
    config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = TransformerLM(config).to(device)
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    # Optimizer & scheduler
    param_groups = build_adamw_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))  # Use standard LLM beta values
    scheduler = create_cosine_lr_scheduler(optimizer, args.warmup_steps, args.max_steps, args.min_lr_ratio)

    # Resume from checkpoint
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from step {start_step}")

    compiled_model = torch.compile(model)

    # Data
    train_loader, val_loader = create_bin_dataloaders(
        args.meta,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Mixed precision: bfloat16 on CUDA, disabled on MPS/CPU
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    )

    ckpt_dir = args.out_dir / args.run_name
    best_val_loss = float("inf")
    train_iter = iter(train_loader)

    model.train()
    t0 = time.perf_counter()

    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps, desc=args.run_name)

    # --- Training loop ---
    for step in pbar:
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with autocast_ctx:
                _, loss = compiled_model(x, y)

            (loss / args.grad_accum_steps).backward()
            loss_accum += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        # Logging
        if (step + 1) % args.log_interval == 0:
            t1 = time.perf_counter()
            ms_per_step = (t1 - t0) / args.log_interval * 1000
            t0 = t1
            lr = scheduler.get_last_lr()[0]
            avg_loss = loss_accum / args.grad_accum_steps
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", ms=f"{ms_per_step:.0f}")

        # Validation
        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
            tqdm.write(f"  step {step + 1} | val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, step + 1, args)
                tqdm.write(f"  saved best checkpoint (val loss {best_val_loss:.4f})")

        # Periodic checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, step + 1, args)
            tqdm.write(f"  saved checkpoint at step {step + 1}")

    # Final checkpoint
    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, args.max_steps, args)
    print(f"Training complete. Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
