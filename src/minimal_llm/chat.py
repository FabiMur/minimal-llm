"""Interactive chat-mode generation for an SFT-finetuned minimal-llm checkpoint."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chat-mode generation."""
    parser = argparse.ArgumentParser(description="Chat with an SFT-finetuned minimal-llm checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to an SFT checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt for the conversation.")
    parser.add_argument("--num_new_tokens", type=int, default=256, help="Max tokens to generate per reply.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Only sample from the top k most likely tokens.")
    return parser.parse_args()
