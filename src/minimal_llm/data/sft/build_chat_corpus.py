"""Build a mixed chat corpus from HuggingFaceH4/no_robots and teknium/OpenHermes-2.5."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import IterableDataset, load_dataset
from tqdm import tqdm

Message = dict[str, str]
Conversation = list[Message]

_ALLOWED_ROLES = {"system", "user", "assistant"}
_OPEN_HERMES_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def is_valid_conversation(messages: Conversation) -> bool:
    """Check that a conversation has content on every turn and ends with an assistant reply.

    Args:
        messages: Normalized role/content messages.

    Returns:
        True if the conversation is non-empty, every turn has content, the last turn is
        from the assistant, and there is at least one user and one assistant turn.
    """
    if not messages:
        return False
    if any(not m["content"] for m in messages):
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return any(m["role"] == "user" for m in messages) and any(m["role"] == "assistant" for m in messages)


def yield_no_robots(ds: IterableDataset) -> Iterator[Conversation]:
    """Yield cleaned conversations from a streamed HuggingFaceH4/no_robots dataset.

    Extracts the "messages" field from each record, strips whitespace, and yields
    conversations that pass `is_valid_conversation`.

    Args:
        ds: Iterable of records where record["messages"] is a list of role/content dicts.

    Yields:
        Normalized conversations.
    """
    for record in ds:
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue

        conv: Conversation = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role not in _ALLOWED_ROLES or not isinstance(content, str):
                conv = []
                break
            conv.append({"role": role, "content": content.strip()})

        if is_valid_conversation(conv):
            yield conv


def yield_open_hermes(ds: IterableDataset) -> Iterator[Conversation]:
    """Yield cleaned conversations from a streamed teknium/OpenHermes-2.5 dataset.

    Extracts the "conversations" field (ShareGPT-style `from`/`value` turns) from each
    record, maps `from` to a role, strips whitespace, and yields conversations that pass
    `is_valid_conversation`.

    Args:
        ds: Iterable of records where record["conversations"] is a list of from/value dicts.

    Yields:
        Normalized conversations.
    """
    for record in ds:
        turns = record.get("conversations")
        if not isinstance(turns, list):
            continue

        conv: Conversation = []
        for turn in turns:
            role = _OPEN_HERMES_ROLE_MAP.get(turn.get("from"))
            content = turn.get("value")
            if role is None or not isinstance(content, str):
                conv = []
                break
            conv.append({"role": role, "content": content.strip()})

        if is_valid_conversation(conv):
            yield conv


def take_n(elements: Iterable[Conversation], n: int) -> Iterator[Conversation]:
    """Take at most the first n elements from an iterable of conversations.

    Args:
        elements: An iterable producing conversations.
        n: Maximum number of elements to take.

    Yields:
        Up to n elements from the input iterable.
    """
    for i, x in enumerate(elements):
        if i >= n:
            break
        yield x


def build_chat_corpus(
    out_path: Path,
    max_conversations: int,
    r_no_robots: int,
    r_open_hermes: int,
    seed: int,
) -> None:
    """Build a mixed chat corpus and write it to disk.

    Sources:
      - HuggingFaceH4/no_robots (human-written instructions)
      - teknium/OpenHermes-2.5 (ShareGPT-style synthetic conversations)

    Warning: no_robots has fewer than 10K rows. If the requested ratio asks for more
    than that, the function will attempt to fill any shortfall with OpenHermes conversations.

    Args:
        out_path: Output file path where the corpus will be written.
        max_conversations: Target number of conversations to write across both sources.
        r_no_robots: Ratio for no_robots conversations.
        r_open_hermes: Ratio for OpenHermes conversations.
        seed: Random seed used when shuffling streaming datasets.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_r = r_no_robots + r_open_hermes
    n_no_robots = max_conversations * r_no_robots // total_r
    n_open_hermes = max_conversations - n_no_robots

    print(f"no_robots:  {n_no_robots:,} conversations")
    print(f"OpenHermes: {n_open_hermes:,} conversations")
    print("Loading datasets...")

    # Split is not needed for SFT, take "train" as it has all the data
    # Use streaming mode to avoid downloading the full dataset

    no_robots: IterableDataset = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
    open_hermes: IterableDataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    # Shuffle streaming datasets (buffered shuffle)
    no_robots = no_robots.shuffle(buffer_size=10_000, seed=seed)
    open_hermes = open_hermes.shuffle(buffer_size=100_000, seed=seed)

    print("Writing chat corpus...")
    wrote = 0

    with out_path.open("w", encoding="utf-8") as f:
        hermes_iter = yield_open_hermes(open_hermes)
        for messages in tqdm(take_n(yield_no_robots(no_robots), n_no_robots), total=n_no_robots, desc="no_robots"):
            f.write(json.dumps({"messages": messages}) + "\n")
            wrote += 1

        for messages in tqdm(take_n(hermes_iter, n_open_hermes), total=n_open_hermes, desc="OpenHermes"):
            f.write(json.dumps({"messages": messages}) + "\n")
            wrote += 1

        # Fallback: if no_robots ran out early, fill the remainder with OpenHermes
        missing = max_conversations - wrote
        if missing > 0:
            print(f"Filling missing conversations with OpenHermes: {missing:,}")
            for messages in tqdm(take_n(hermes_iter, missing), total=missing, desc="OpenHermes (fill)"):
                f.write(json.dumps({"messages": messages}) + "\n")
                wrote += 1

    file_size_mb = out_path.stat().st_size / (1024**2)
    print(f"Saved {wrote:,} conversations ({file_size_mb:.1f} MB) -> {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/chat_corpus.jsonl", help="Output chat corpus file path.")
    ap.add_argument("--max_conversations", type=int, default=110_000, help="Max conversations to write.")
    ap.add_argument("--ratio-no-robots", type=int, default=1, help="no_robots ratio (e.g. 1).")
    ap.add_argument("--ratio-open-hermes", type=int, default=10, help="OpenHermes ratio (e.g. 10).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the chat corpus builder CLI.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).
    """
    args = parse_args(argv)
    build_chat_corpus(
        out_path=Path(args.out),
        max_conversations=args.max_conversations,
        r_no_robots=args.ratio_no_robots,
        r_open_hermes=args.ratio_open_hermes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
