"""Data loading utilities for binary token files."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_meta(meta_path: str | Path) -> dict:
    """Load metadata JSON produced by tokenize_to_bin.

    Args:
        meta_path: Path to meta.json.

    Returns:
        Parsed metadata dictionary.
    """
    with Path(meta_path).open(encoding="utf-8") as f:
        return json.load(f)


class BinTokenDataset(Dataset):
    """Map-style dataset that serves sliding windows from a binary token file.

    Uses np.memmap for memory-efficient, random-access reads without loading the entire file into RAM.
    """

    def __init__(self, bin_path: str | Path, dtype: str, context_length: int, stride: int | None = None) -> None:
        """Initialize the dataset.

        Args:
            bin_path: Path to the binary token file.
            dtype: NumPy dtype string stored in meta.json (e.g. "uint16").
            context_length: Number of tokens per input sequence.
            stride: Step size between consecutive windows.
                Defaults to context_length (no overlap).
        """
        self.context_length = context_length

        # Defatult to non-overlapping windows
        self.stride = stride or context_length

        # Memory-map the file (read-only, zero-copy, on-demand loading)
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")

        # Each sample needs context_length + 1 tokens (input + target)
        # Substract context_length to ensure the last window has enough tokens for input and target
        self.n_windows = max(0, (len(self.data) - context_length) // self.stride)

    def __len__(self) -> int:
        """Return the number of sliding windows in the dataset."""
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) token tensors for the given window index."""
        start = idx * self.stride
        end = start + self.context_length + 1

        raw = self.data[start:end]  # Memmap slice → NumPy array
        tokens = raw.astype(np.int64)  # Cast to int64, needed for PyTorch embedding layers
        chunk = torch.from_numpy(tokens)  # NumPy Array → PyTorch Tensor

        x = chunk[0 : self.context_length]  # input tokens
        y = chunk[1 : self.context_length + 1]  # target tokens (shifted by 1)
        return x, y
