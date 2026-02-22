"""Build a mixed text corpus from multiple sources.

This module provides functionality to create a combined corpus from three sources:
- WikiText-103 (wikitext-103-raw-v1)
- FineWeb sample (HuggingFaceFW/fineweb, sample-10BT)
- TinyStories (roneneldan/TinyStories)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


def yield_text(ds: IterableDataset) -> Iterator[str]:
    """Yield cleaned text lines from an iterable dataset.

    This function extracts the "text" field from each record, normalizes whitespace,
    and yields non-empty lines.

    Args:
        ds: Iterable of records where record["text"] contains the text.

    Yields:
        Cleaned text lines with normalized whitespace.
    """
    for record in ds:
        text = record.get("text")
        if isinstance(text, str):
            text = " ".join(text.split())
            if text:
                yield text


def take_n(elements: Iterable[str], n: int) -> Iterator[str]:
    """Take at most the first n elements from an iterable of strings.

    Args:
        elements: An iterable producing strings.
        n: Maximum number of elements to take.

    Yields:
        Up to n elements from the input iterable.
    """
    for i, x in enumerate(elements):
        if i >= n:
            break
        yield x


def build_corpus(
    out_path: Path,
    max_lines: int,
    r_wiki: int,
    r_web: int,
    r_stories: int,
    seed: int,
) -> None:
    """Build a mixed text corpus and write it to disk.

    Sources:
      - WikiText-103 (wikitext-103-raw-v1)
      - FineWeb sample (HuggingFaceFW/fineweb, sample-10BT)
      - TinyStories (roneneldan/TinyStories)

    Args:
        out_path: Output file path where the corpus will be written.
        max_lines: Maximum number of lines to write across all sources.
        r_wiki: Ratio for Wikipedia lines.
        r_web: Ratio for Web lines.
        r_stories: Ratio for Stories lines.
        seed: Random seed used when shuffling streaming datasets.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_r = r_wiki + r_web + r_stories
    n_wiki = max_lines * r_wiki // total_r
    n_web = max_lines * r_web // total_r
    n_st = max_lines - n_wiki - n_web

    print(f"Wikipedia: {n_wiki:,} lines")
    print(f"Web: {n_web:,} lines")
    print(f"Stories: {n_st:,} lines")
    print("Loading datasets...")

    # Split is not needed for LLMs, take "train" as it has most of the data
    # Use streaming mode to avoid downloading the full dataset

    wiki: IterableDataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    web: IterableDataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    stories: IterableDataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Shuffle streaming datasets (buffered shuffle)
    wiki = wiki.shuffle(buffer_size=100_000, seed=seed)
    web = web.shuffle(buffer_size=100_000, seed=seed)
    stories = stories.shuffle(buffer_size=10_000, seed=seed)

    print("Writing corpus...")
    wrote = 0

    with out_path.open("w", encoding="utf-8") as f:
        for text in tqdm(take_n(yield_text(wiki), n_wiki), total=n_wiki, desc="Wiki"):
            f.write(text + "\n")
            wrote += 1

        for text in tqdm(take_n(yield_text(web), n_web), total=n_web, desc="Web"):
            f.write(text + "\n")
            wrote += 1

        for text in tqdm(take_n(yield_text(stories), n_st), total=n_st, desc="Stories"):
            f.write(text + "\n")
            wrote += 1

    file_size_gb = out_path.stat().st_size / (1024**3)
    print(f"Saved {wrote:,} lines ({file_size_gb:.2f} GB) -> {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt", help="Output corpus file path.")
    ap.add_argument("--max_lines", type=int, default=10_000_000, help="Max lines to write.")
    ap.add_argument("--ratio-wiki", type=int, default=1, help="Wiki ratio (e.g. 2).")
    ap.add_argument("--ratio-web", type=int, default=1, help="Web ratio (e.g. 2).")
    ap.add_argument("--ratio-stories", type=int, default=1, help="Stories ratio (e.g. 2).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the corpus builder CLI.

    Args:
        argv: Optional argument vector (defaults to sys.argv if None).
    """
    args = parse_args(argv)
    build_corpus(
        out_path=Path(args.out),
        max_lines=args.max_lines,
        r_wiki=args.ratio_wiki,
        r_web=args.ratio_web,
        r_stories=args.ratio_stories,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
