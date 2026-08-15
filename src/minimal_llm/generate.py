"""Language model inference script."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from minimal_llm.model import ModelConfig, TransformerLM
from minimal_llm.train import get_device


def load_model(ckpt_path: Path, device: torch.device) -> TransformerLM:
    """Rebuild a model from a training checkpoint's saved args and load its weights.

    Args:
        ckpt_path: Path to a checkpoint saved by train.py.
        device: Device to load the model onto.

    Returns:
        The model in eval mode, ready for inference.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    config = ModelConfig(
        vocab_size=train_args["vocab_size"],
        context_length=train_args["context_length"],
        d_model=train_args["d_model"],
        n_layers=train_args["n_layers"],
        n_heads=train_args["n_heads"],
        n_kv_heads=train_args["n_kv_heads"],
        rope_theta=train_args["rope_theta"],
    )
    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    num_new_tokens: int,
    temperature: float,
    top_k: int | None,
    add_bos: bool,
) -> str:
    """Encode a prompt, run generation, and decode the result back to text.

    Args:
        model: Loaded language model.
        tokenizer: Tokenizer used to encode/decode text.
        prompt: Prompt text to generate from.
        device: Device to run generation on.
        num_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Only sample from the top k most likely tokens.
        add_bos: Prepend the <|bos|> token to the prompt before encoding.

    Returns:
        The decoded text (including the prompt).
    """
    ids = tokenizer.encode(prompt).ids
    if add_bos:
        bos_id = tokenizer.token_to_id("<|bos|>")
        if bos_id is not None:
            ids = [bos_id] + ids

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, num_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def parse_args() -> argparse.Namespace:
    """Parse cli arguments for generation."""
    parser = argparse.ArgumentParser(description="Generate text from a trained minimal-llm checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a checkpoint saved by train.py.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text. If omitted, starts an interactive loop.")
    parser.add_argument("--num_new_tokens", type=int, default=200, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Only sample from the top k most likely tokens.")
    parser.add_argument("--no_bos", action="store_true", help="Don't prepend the <|bos|> token to the prompt.")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and generate text, from a prompt or in an interactive loop."""
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def run(prompt: str) -> None:
        text = generate_text(
            model,
            tokenizer,
            prompt,
            device,
            num_new_tokens=args.num_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            add_bos=not args.no_bos,
        )
        print(text)

    if args.prompt is not None:
        run(args.prompt)
        return

    print("Interactive mode. Enter a prompt (empty line to quit).")
    while True:
        try:
            prompt = input("\n> ")
        except EOFError:
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
