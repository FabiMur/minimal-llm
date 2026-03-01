"""Tokenize a text corpus and save token IDs to binary files for training."""

import argparse
import json
from pathlib import Path
from typing import IO

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def write_ids(
    fh: IO[bytes],
    ids: list[int],
    dtype: type[np.uint16] | type[np.int32],
) -> int:
    """Write token IDs to a binary file.

    Converts a list of token IDs to a NumPy array with the specified dtype
    and appends it to an open binary file handle.

    Args:
        fh: File handle opened in binary write mode.
        ids: Sequence of token IDs.
        dtype: NumPy dtype used to store the IDs.

    Returns:
        Number of token IDs written.
    """
    if not ids:
        return 0

    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(fh)
    return arr.size


def flush_encoded(
    encoded_batch: list,
    train_file: IO[bytes],
    val_file: IO[bytes],
    rng: np.random.Generator,
    val_ratio: float,
    dtype: type[np.uint16] | type[np.int32],
    eos_id: int | None,
    bos_id: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Write encoded sequences to train/validation splits.

    Args:
        encoded_batch: List of encoded tokenizer outputs.
        train_file: Binary file handle for training split.
        val_file: Binary file handle for validation split.
        rng: Random number generator.
        val_ratio: Probability of assigning a line to validation.
        dtype: NumPy dtype for storage.
        eos_id: Optional EOS token ID.
        bos_id: Optional BOS token ID.

    Returns:
        Tuple containing:
            total_tokens,
            train_tokens,
            val_tokens,
            train_lines,
            val_lines
    """
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0
    train_lines = 0
    val_lines = 0

    for enc in encoded_batch:
        ids = enc.ids
        if not ids:
            continue

        ids_to_write = ([bos_id] if bos_id is not None else []) + ids + ([eos_id] if eos_id is not None else [])
        total_tokens += len(ids_to_write)

        if rng.random() < val_ratio:
            val_tokens += write_ids(val_file, ids_to_write, dtype)
            val_lines += 1
        else:
            train_tokens += write_ids(train_file, ids_to_write, dtype)
            train_lines += 1

    return total_tokens, train_tokens, val_tokens, train_lines, val_lines


def main() -> None:
    """Tokenize a text corpus and save token IDs to binary files.

    Performs per-line train/validation splitting and stores token IDs
    in compact binary format.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--corpus", type=str, default="data/corpus.txt")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--train_name", type=str, default="train.bin")
    ap.add_argument("--val_name", type=str, default="val.bin")
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--batch_lines", type=int, default=10_000)
    ap.add_argument("--dtype", type=str, choices=["uint16", "int32"], default="uint16")
    ap.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True)  # --add_eos / --no-add_eos flag
    args = ap.parse_args()

    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val_ratio must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    if args.dtype == "uint16":
        if vocab_size > 65535:
            raise ValueError(f"vocab_size={vocab_size} doesn't fit in uint16.")
        dtype = np.uint16
    else:
        dtype = np.int32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    meta_path = out_dir / "meta.json"

    eos_id = tok.token_to_id("[EOS]") if args.add_eos else None
    if args.add_eos and eos_id is None:
        raise ValueError("Passed --add_eos but tokenizer has no [EOS] token.")

    bos_id = tok.token_to_id("[BOS]") if args.add_eos else None
    if args.add_eos and bos_id is None:
        raise ValueError("Passed --add_eos but tokenizer has no [BOS] token.")

    total_lines = 0
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0
    split_train_lines = 0
    split_val_lines = 0

    print(f"Tokenizer: {args.tokenizer} (vocab_size={vocab_size})")
    print(f"Corpus:    {args.corpus}")
    print(f"Output:    {train_path} / {val_path} (dtype={args.dtype}, val_ratio={args.val_ratio}, seed={args.seed})")

    batch: list[str] = []

    with (
        open(args.corpus, encoding="utf-8") as file,
        open(train_path, "wb") as train_file,
        open(val_path, "wb") as val_file,
    ):
        # Determine total lines for progress bar
        if args.max_lines is not None:
            total_for_bar = args.max_lines
        else:
            print("Counting total lines...")
            with open(args.corpus, encoding="utf-8") as tmp_f:
                total_for_bar = sum(1 for _ in tmp_f)

        iterator = tqdm(
            file,
            total=total_for_bar,
            desc="Tokenizing",
            unit="lines",
            dynamic_ncols=True,
        )

        for line in iterator:
            if args.max_lines is not None and total_lines >= args.max_lines:
                break

            line = line.strip()

            if not line:
                continue

            total_lines += 1

            batch.append(line)

            if len(batch) >= args.batch_lines:
                enc = tok.encode_batch(batch)
                batch = []

                dt, tr_tok, v_tok, tr_lines, v_lines = flush_encoded(
                    enc,
                    train_file,
                    val_file,
                    rng,
                    args.val_ratio,
                    dtype,
                    eos_id,
                    bos_id,
                )

                total_tokens += dt
                train_tokens += tr_tok
                val_tokens += v_tok
                split_train_lines += tr_lines
                split_val_lines += v_lines

        if batch:
            enc = tok.encode_batch(batch)
            dt, tr_tok, v_tok, tr_lines, v_lines = flush_encoded(
                enc,
                train_file,
                val_file,
                rng=rng,
                val_ratio=args.val_ratio,
                dtype=dtype,
                eos_id=eos_id,
                bos_id=bos_id,
            )

            total_tokens += dt
            train_tokens += tr_tok
            val_tokens += v_tok
            split_train_lines += tr_lines
            split_val_lines += v_lines

    train_to_val_token_ratio = train_tokens / val_tokens if val_tokens > 0 else float("inf")

    meta = {
        "tokenizer": str(args.tokenizer),
        "corpus": str(args.corpus),
        "vocab_size": vocab_size,
        "dtype": args.dtype,
        "np_dtype": np.dtype(dtype).name,
        "token_bytes": int(np.dtype(dtype).itemsize),
        "add_eos": bool(args.add_eos),
        "eos_id": eos_id,
        "bos_id": bos_id,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_lines": int(split_train_lines),
        "val_lines": int(split_val_lines),
        "total_lines": int(total_lines),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "total_tokens": int(total_tokens),
        "train_to_val_token_ratio": train_to_val_token_ratio,
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"Lines read:    {total_lines:,}")
    print(f"Tokens total:  {total_tokens:,}")
    print(f"Train tokens:  {train_tokens:,}")
    print(f"Val tokens:    {val_tokens:,}")
    print(f"Train/Val token ratio: {train_to_val_token_ratio:.2f}")
    print(f"Meta saved:    {meta_path}")


if __name__ == "__main__":
    main()
