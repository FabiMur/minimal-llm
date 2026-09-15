"""Supervised fine-tuning loop for a pretrained minimal-llm checkpoint."""

import argparse
from datetime import date
from pathlib import Path

import torch

from minimal_llm.data.sft.data_loaders import create_chat_dataloaders
from minimal_llm.generate import load_model
from minimal_llm.train import (
    build_adamw_param_groups,
    create_cosine_lr_scheduler,
    get_device,
    load_checkpoint,
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
