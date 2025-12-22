# src/data.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class BinDatasetConfig:
    train_bin: Path
    val_bin: Path
    np_dtype: str  # "uint16" or "int32"
    vocab_size: int
    add_eos: bool
    eos_id: Optional[int]


def load_meta(meta_path: str | Path) -> BinDatasetConfig:
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    base = meta_path.parent
    train_bin = (base / meta["train_bin"]).resolve()
    val_bin = (base / meta["val_bin"]).resolve()

    np_dtype = meta.get("np_dtype", meta.get("dtype"))
    if np_dtype == "uint16":
        np_dtype = "uint16"
    elif np_dtype == "int32":
        np_dtype = "int32"
    else:
        raise ValueError(f"Unsupported np_dtype in meta.json: {np_dtype}")

    return BinDatasetConfig(
        train_bin = train_bin,
        val_bin = val_bin,
        np_dtype = np_dtype,
        vocab_size = int(meta["vocab_size"]),
        add_eos = bool(meta.get("add_eos", False)),
        eos_id = meta.get("eos_id"),
    )


class BinTokenDataset(Dataset):
    """
    Dataset over a flat token stream stored in a .bin file.

    It returns (x, y) where:
      x = tokens[i : i+T]
      y = tokens[i+1 : i+T+1]
    """

    def __init__(
        self,
        bin_path: str | Path,
        np_dtype: str,
        context_length: int,
        stride: Optional[int] = None,
    ):
        self.bin_path = Path(bin_path)
        self.context_length = int(context_length)
        self.stride = int(stride) if stride is not None else self.context_length

        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        if np_dtype not in ("uint16", "int32"):
            raise ValueError("np_dtype must be 'uint16' or 'int32'")
        self.np_dtype = np_dtype

        # Memory-map token file (lazy loading via OS paging)
        self.tokens = np.memmap(self.bin_path, dtype = self.np_dtype, mode = "r")

        # Need T+1 tokens to create (x, y)
        needed = self.context_length + 1
        if len(self.tokens) < needed:
            self.num_windows = 0
        else:
            self.num_windows = 1 + (len(self.tokens) - needed) // self.stride

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError("Index out of range")

        start = idx * self.stride
        end = start + self.context_length + 1

        # Get token window from memmap (array slice, no copy)
        window = self.tokens[start:end]  # dtype uint16/int32
        x = torch.from_numpy(window[:-1].astype(np.int64, copy = False))
        y = torch.from_numpy(window[1:].astype(np.int64, copy = False))

        return x, y


def create_bin_dataloaders(
    meta_path: str | Path,
    context_length: int = 1024,
    stride: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    cfg = load_meta(meta_path)

    train_ds = BinTokenDataset(
        bin_path = cfg.train_bin,
        np_dtype = cfg.np_dtype,
        context_length = context_length,
        stride = stride,
    )
    val_ds = BinTokenDataset(
        bin_path = cfg.val_bin,
        np_dtype = cfg.np_dtype,
        context_length = context_length,
        stride = stride,
    )

    use_persistent = num_workers > 0
    loader_kwargs = dict(
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory and torch.cuda.is_available(),
        persistent_workers = use_persistent,
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle = True,
        drop_last = True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle = False,
        drop_last = False,
        **loader_kwargs,
    )
    return train_loader, val_loader
