"""Data loaders for SFT chat training examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChatBinDataset(Dataset):
    """Map-style dataset that serves sliding windows from paired ids/labels binary files.

    `ids` holds ChatML token ids for every conversation written back-to-back, while
    `labels` is the token-aligned loss target for the *next* token (see
    `tokenize_chat.encode_conversation`), with `IGNORE_INDEX` wherever that next token
    must not contribute to the loss (everything but assistant output). A window's input
    is read from `ids` unshifted and its target from `labels` shifted by one, so windows
    can straddle conversation boundaries without mixing up which stream to shift.

    Uses np.memmap for memory-efficient, random-access reads without loading either file
    into RAM.
    """

    def __init__(
        self,
        ids_path: str | Path,
        labels_path: str | Path,
        ids_dtype: str,
        labels_dtype: str,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            ids_path: Path to the binary token ids file.
            labels_path: Path to the binary loss-label file, aligned with `ids_path`.
            ids_dtype: NumPy dtype string for the ids file (e.g. "uint16").
            labels_dtype: NumPy dtype string for the labels file (e.g. "int32").
            context_length: Number of tokens per input sequence.
            stride: Step size between consecutive windows. Defaults to context_length.
        """
        self.context_length = context_length
        self.stride = stride or context_length

        self.ids = np.memmap(ids_path, dtype=np.dtype(ids_dtype), mode="r")
        self.labels = np.memmap(labels_path, dtype=np.dtype(labels_dtype), mode="r")
        if len(self.ids) != len(self.labels):
            raise ValueError(f"ids/labels length mismatch: {len(self.ids)} != {len(self.labels)}")

        # Each sample needs context_length + 1 tokens (input + shifted target)
        self.n_windows = max(0, (len(self.ids) - context_length) // self.stride)

    def __len__(self) -> int:
        """Return the number of sliding windows in the dataset."""
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) token tensors for the given window index."""
        start = idx * self.stride

        x = self.ids[start : start + self.context_length].astype(np.int64)
        y = self.labels[start + 1 : start + self.context_length + 1].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
