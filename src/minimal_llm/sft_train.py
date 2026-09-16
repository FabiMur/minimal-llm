"""Supervised fine-tuning loop for a pretrained minimal-llm checkpoint."""

import argparse
import contextlib
import time
from datetime import date
from pathlib import Path

import torch
from tqdm import tqdm

from minimal_llm.data.sft.data_loaders import create_chat_dataloaders
from minimal_llm.generate import load_model
from minimal_llm.train import (
    build_adamw_param_groups,
    create_cosine_lr_scheduler,
    evaluate,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SFT."""
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained minimal-llm checkpoint on chat data.")

    # IO
    parser.add_argument("--run_name", type=str, default=str(date.today().isoformat()), help="SFT run name.")
    parser.add_argument("--init_checkpoint", type=Path, required=True, help="Pretrained checkpoint to fine-tune.")
    parser.add_argument("--meta", type=Path, default=Path("artifacts/meta_chat.json"), help="Path to meta_chat.json.")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/checkpoints_sft"), help="Checkpoint dir.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to an in-progress SFT checkpoint to resume.")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--no_grad_checkpoint", action="store_true", help="Disable gradient checkpointing.")

    # Evaluation and logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log loss every N steps.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Run validation every N steps.")
    parser.add_argument("--eval_batches", type=int, default=50, help="Max batches per validation run.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N steps.")

    return parser.parse_args()


def setup(args: argparse.Namespace) -> tuple:
    """Build the model, optimizer, scheduler, and dataloaders for an SFT run.

    Args:
        args: Parsed SFT CLI arguments.

    Returns:
        Tuple of (model, optimizer, scheduler, train_loader, val_loader, device, start_step).
    """
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.init_checkpoint, device, grad_checkpoint=not args.no_grad_checkpoint)
    model.train()
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    param_groups = build_adamw_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scheduler = create_cosine_lr_scheduler(optimizer, args.warmup_steps, args.max_steps, args.min_lr_ratio)

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from step {start_step}")

    train_loader, val_loader = create_chat_dataloaders(
        args.meta,
        context_length=model.config.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return model, optimizer, scheduler, train_loader, val_loader, device, start_step


def main() -> None:
    """Run the SFT loop."""
    args = parse_args()
    model, optimizer, scheduler, train_loader, val_loader, device, start_step = setup(args)

    compiled_model = model if args.no_compile else torch.compile(model)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    )

    ckpt_dir = args.out_dir / args.run_name
    best_val_loss = float("inf")
    train_iter = iter(train_loader)

    model.train()
    t0 = time.perf_counter()

    pbar = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps, desc=args.run_name)

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

        if (step + 1) % args.log_interval == 0:
            t1 = time.perf_counter()
            ms_per_step = (t1 - t0) / args.log_interval * 1000
            t0 = t1
            lr = scheduler.get_last_lr()[0]
            avg_loss = loss_accum / args.grad_accum_steps
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", ms=f"{ms_per_step:.0f}")

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
            tqdm.write(f"  step {step + 1} | val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, step + 1, args)
                tqdm.write(f"  saved best checkpoint (val loss {best_val_loss:.4f})")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, step + 1, args)
            tqdm.write(f"  saved checkpoint at step {step + 1}")

    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, args.max_steps, args)
    print(f"SFT complete. Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
