"""Tokenize chat conversations into ChatML-formatted training examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import IO

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

IGNORE_INDEX = -1


def encode_message(tok: Tokenizer, role: str, content: str, is_target: bool) -> tuple[list[int], list[int]]:
    """Encode one ChatML turn into token ids and aligned loss labels.

    Wraps the turn as "<|im_start|>{role}", a newline, "{content}<|im_end|>", and a trailing
    newline. When `is_target` is True (assistant turns), labels mirror the ids for the content
    and closing delimiter; everywhere else labels are `IGNORE_INDEX`, so the loss only trains
    on assistant output.

    Args:
        tok: Loaded BPE tokenizer with ChatML special tokens.
        role: Message role ("system", "user", or "assistant").
        content: Message text content.
        is_target: Whether this turn's tokens should count toward the loss.

    Returns:
        Tuple of (ids, labels), the same length.
    """
    im_start_id = tok.token_to_id("<|im_start|>")
    im_end_id = tok.token_to_id("<|im_end|>")

    header_ids = [im_start_id, *tok.encode(role + "\n").ids]
    body_ids = tok.encode(content).ids
    footer_ids = [im_end_id, *tok.encode("\n").ids]

    ids = header_ids + body_ids + footer_ids
    if is_target:
        labels = [IGNORE_INDEX] * len(header_ids) + body_ids + footer_ids
    else:
        labels = [IGNORE_INDEX] * len(ids)
    return ids, labels


def encode_conversation(
    tok: Tokenizer, messages: list[dict[str, str]], bos_id: int | None
) -> tuple[list[int], list[int]]:
    """Encode a full conversation into token ids and aligned loss labels.

    Concatenates each message's ChatML turn (see `encode_message`) in order, optionally
    prefixed with a BOS token that is always masked out of the loss.

    Args:
        tok: Loaded BPE tokenizer with ChatML special tokens.
        messages: Ordered role/content messages, e.g. `[{"role": ..., "content": ...}, ...]`.
        bos_id: Optional BOS token id to prepend.

    Returns:
        Tuple of (ids, labels), the same length.
    """
    ids: list[int] = [bos_id] if bos_id is not None else []
    labels: list[int] = [IGNORE_INDEX] if bos_id is not None else []

    for message in messages:
        turn_ids, turn_labels = encode_message(
            tok, message["role"], message["content"], is_target=message["role"] == "assistant"
        )
        ids.extend(turn_ids)
        labels.extend(turn_labels)

    return ids, labels


def write_ids(fh: IO[bytes], ids: list[int], dtype: type[np.uint16] | type[np.int32]) -> int:
    """Write token ids or labels to a binary file.

    Converts a list of ints to a NumPy array with the specified dtype and appends it to
    an open binary file handle.

    Args:
        fh: File handle opened in binary write mode.
        ids: Sequence of token ids or labels.
        dtype: NumPy dtype used to store the values.

    Returns:
        Number of values written.
    """
    if not ids:
        return 0

    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(fh)
    return arr.size


def tokenize_chat_corpus(
    corpus_path: Path,
    tokenizer_path: Path,
    tok: Tokenizer,
    out_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict:
    """Tokenize a chat corpus into train/val ids and label binaries.

    Reads one JSON conversation per line (as written by `build_chat_corpus.py`), encodes
    it with `encode_conversation`, and writes ids (uint16) and labels (int32) to separate
    binary files, split per conversation into train/val.

    Args:
        corpus_path: Path to the chat corpus JSONL file.
        tokenizer_path: Path to the tokenizer file `tok` was loaded from (recorded in meta).
        tok: Loaded BPE tokenizer with ChatML special tokens.
        out_dir: Directory to write the output binaries into.
        val_ratio: Probability of assigning a conversation to validation.
        seed: Random seed used for the train/val split.

    Returns:
        Metadata dictionary describing the produced files and token/conversation counts.
    """
    bos_id = tok.token_to_id("<|bos|>")
    rng = np.random.default_rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids_path = out_dir / "train_ids.bin"
    train_labels_path = out_dir / "train_labels.bin"
    val_ids_path = out_dir / "val_ids.bin"
    val_labels_path = out_dir / "val_labels.bin"

    train_conversations = 0
    val_conversations = 0
    train_tokens = 0
    val_tokens = 0

    with (
        corpus_path.open(encoding="utf-8") as corpus_file,
        train_ids_path.open("wb") as train_ids_file,
        train_labels_path.open("wb") as train_labels_file,
        val_ids_path.open("wb") as val_ids_file,
        val_labels_path.open("wb") as val_labels_file,
    ):
        for line in tqdm(corpus_file, desc="Tokenizing chat corpus", unit="conversations"):
            messages = json.loads(line)["messages"]
            ids, labels = encode_conversation(tok, messages, bos_id)

            if rng.random() < val_ratio:
                write_ids(val_ids_file, ids, np.uint16)
                write_ids(val_labels_file, labels, np.int32)
                val_conversations += 1
                val_tokens += len(ids)
            else:
                write_ids(train_ids_file, ids, np.uint16)
                write_ids(train_labels_file, labels, np.int32)
                train_conversations += 1
                train_tokens += len(ids)

    return {
        "tokenizer": str(tokenizer_path),
        "corpus": str(corpus_path),
        "vocab_size": tok.get_vocab_size(),
        "ids_dtype": "uint16",
        "labels_dtype": "int32",
        "ignore_index": IGNORE_INDEX,
        "bos_id": bos_id,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_conversations": train_conversations,
        "val_conversations": val_conversations,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_ids_bin": str(train_ids_path),
        "train_labels_bin": str(train_labels_path),
        "val_ids_bin": str(val_ids_path),
        "val_labels_bin": str(val_labels_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="artifacts/chat_corpus.jsonl", help="Path to the chat corpus JSONL file.")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json", help="Path to the trained tokenizer.")
    ap.add_argument("--out_dir", default="artifacts", help="Directory to write the output binaries into.")
    ap.add_argument("--val_ratio", type=float, default=0.01, help="Fraction of conversations held out for val.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the chat corpus tokenizer CLI.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).
    """
    args = parse_args(argv)
    tok = Tokenizer.from_file(args.tokenizer)

    meta = tokenize_chat_corpus(
        corpus_path=Path(args.corpus),
        tokenizer_path=Path(args.tokenizer),
        tok=tok,
        out_dir=Path(args.out_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    meta_path = Path(args.out_dir) / "meta_chat.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Train: {meta['train_conversations']:,} conversations, {meta['train_tokens']:,} tokens")
    print(f"Val:   {meta['val_conversations']:,} conversations, {meta['val_tokens']:,} tokens")
    print(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
