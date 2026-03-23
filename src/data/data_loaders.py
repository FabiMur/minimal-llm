"""Data loading utilities for binary token files."""

import json
from pathlib import Path


def load_meta(meta_path: str | Path) -> dict:
    """Load metadata JSON produced by tokenize_to_bin.

    Args:
        meta_path: Path to meta.json.

    Returns:
        Parsed metadata dictionary.
    """
    with Path(meta_path).open(encoding="utf-8") as f:
        return json.load(f)
