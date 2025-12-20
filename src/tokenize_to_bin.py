import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def write_ids(fh, ids, dtype):
    """
    Write a list of token IDs to a binary file as a numpy array.
    Args:
        fh: File handle opened in binary write mode
        ids: List of token IDs (integers)
        dtype: Numpy data type (e.g., np.uint16, np.int32)
    """
    if not ids:
        return 0
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(fh)
    return arr.size


def main():
    """
    Tokenize a text corpus and save token IDs to binary files.
    Reads lines from corpus, tokenizes with the given tokenizer, and writes to train/val bins.
    Split is done per line.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--corpus", type=str, default="data/corpus.txt")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--train_name", type=str, default="train.bin")
    ap.add_argument("--val_name", type=str, default="val.bin")
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducible train/val split")
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--batch_lines", type=int, default=10_000)
    ap.add_argument("--dtype", type=str, choices=["uint16", "int32"], default="uint16")
    ap.add_argument("--add_eos", action="store_true", help="Append EOS token after each line")
    args = ap.parse_args()

    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val_ratio must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    # data type selection
    if args.dtype == "uint16":
        if vocab_size > 65535:
            raise ValueError(f"vocab_size={vocab_size} doesn't fit in uint16.")
        dtype = np.uint16
    else:
        dtype = np.int32

    # prepare output files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    meta_path = out_dir / "meta.json"

    # setup EOS  token if specified
    eos_id = tok.token_to_id("[EOS]") if args.add_eos else None
    if args.add_eos and eos_id is None:
        raise ValueError("Passed --add_eos but tokenizer has no [EOS] token.")

    total_lines = 0
    total_tokens = 0
    train_tokens = 0
    val_tokens = 0
    split_train_lines = 0
    split_val_lines = 0

    print(f"Tokenizer: {args.tokenizer} (vocab_size={vocab_size})")
    print(f"Corpus:    {args.corpus}")
    print(
        f"Output:    {train_path} / {val_path}  "
        f"(dtype={args.dtype}, val_ratio={args.val_ratio}, seed={args.seed})"
    )

    def flush_encoded(encoded_batch, train_f, val_f):
        nonlocal total_tokens, train_tokens, val_tokens, split_train_lines, split_val_lines

        for enc in encoded_batch:
            ids = enc.ids
            if not ids:
                continue

            # Optional EOS per line
            if eos_id is not None:
                ids_to_write = ids + [eos_id]
            else:
                ids_to_write = ids

            n = len(ids_to_write)
            total_tokens += n

            # Split per line
            if rng.random() < args.val_ratio:
                val_tokens += write_ids(val_f, ids_to_write, dtype)
                split_val_lines += 1
            else:
                train_tokens += write_ids(train_f, ids_to_write, dtype)
                split_train_lines += 1

    batch = []
    with open(args.corpus, "r", encoding="utf-8") as f, \
         open(train_path, "wb") as train_f, \
         open(val_path, "wb") as val_f:

        for line in f:
            if args.max_lines is not None and total_lines >= args.max_lines:
                break

            line = line.strip()
            total_lines += 1
            if not line:
                continue

            batch.append(line)

            if len(batch) >= args.batch_lines:
                enc = tok.encode_batch(batch)
                flush_encoded(enc, train_f, val_f)
                batch = []
                print(
                    f"lines={total_lines:,} tokens={total_tokens:,} "
                    f"(train={train_tokens:,} val={val_tokens:,} | "
                    f"train_lines={split_train_lines:,} val_lines={split_val_lines:,})"
                )

        # if last batch not empty, flush it
        if batch:
            enc = tok.encode_batch(batch)
            flush_encoded(enc, train_f, val_f)

    # Save meta data
    meta = {
        "tokenizer": str(args.tokenizer),
        "corpus": str(args.corpus),
        "vocab_size": vocab_size,
        "dtype": args.dtype,
        "np_dtype": np.dtype(dtype).name,
        "token_bytes": int(np.dtype(dtype).itemsize),
        "context_hint": None,
        "add_eos": bool(args.add_eos),
        "eos_id": eos_id,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "lines_read": total_lines,
        "train_lines": int(split_train_lines),
        "val_lines": int(split_val_lines),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "total_tokens": int(total_tokens),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done")
    print(f"Lines read:    {total_lines:,}")
    print(f"Tokens total:  {total_tokens:,}")
    print(f"Train tokens:  {train_tokens:,}")
    print(f"Val tokens:    {val_tokens:,}")
    print(f"Train lines:   {split_train_lines:,}")
    print(f"Val lines:     {split_val_lines:,}")
    print(f"Meta saved:    {meta_path}")


if __name__ == "__main__":
    main()
