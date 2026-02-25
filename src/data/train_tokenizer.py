"""Train a BPE tokenizer on the given corpus file."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def main() -> None:
    """Train a BPE tokenizer on the given corpus file."""
    ap = argparse.ArgumentParser(description="Train BPE tokenizer on corpus")
    ap.add_argument("--corpus", default="data/corpus.txt", help="Path to input corpus file")
    ap.add_argument("--vocab_size", type=int, default=32_000, help="Vocabulary size for the tokenizer")
    ap.add_argument("--out", default="artifacts/tokenizer.json", help="Path to save trained tokenizer")
    ap.add_argument("--min_frequency", type=int, default=5, help="Minimum token frequency")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["[PAD]", "[BOS]", "[EOS]"],
    )

    tok.train(files=[args.corpus], trainer=trainer)

    tok.save(str(out_path))
    print(f"Saved tokenizer -> {out_path}")


if __name__ == "__main__":
    main()
