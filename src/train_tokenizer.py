from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Train BPE tokenizer on corpus")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--out", default="artifacts/tokenizer.json")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and configure the tokenizer
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"]
    )

    # Train tokenizer
    tok.train(files=[args.corpus], trainer=trainer)

    # Post-processing to add special tokens
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)

    # Save trained tokenizer
    tok.save(args.out)
    print(f"Saved tokenizer -> {args.out}")

if __name__ == "__main__":
    main()
