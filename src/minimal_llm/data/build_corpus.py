"""Build a mixed text corpus from multiple sources.

This module provides functionality to create a combined corpus from four sources:
- Wikipedia (wikimedia/wikipedia, 20231101.en)
- FineWeb sample (HuggingFaceFW/fineweb, sample-10BT)
- TinyStories (roneneldan/TinyStories)
- OpenWebText (Skylion007/openwebtext)
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
    r_fine_web: int,
    r_stories: int,
    r_open_web: int,
    seed: int,
) -> None:
    """Build a mixed text corpus and write it to disk.

    Sources:
      - Wikipedia (wikimedia/wikipedia, 20231101.en)
      - FineWeb sample (HuggingFaceFW/fineweb, sample-10BT)
      - TinyStories (roneneldan/TinyStories)
      - OpenWebText (Skylion007/openwebtext)

    Warning: Some datasets may have fewer lines than requested.
    The function will attempt to fill any shortfall with FineWeb lines.

    Args:
        out_path: Output file path where the corpus will be written.
        max_lines: Target number of lines to write across all sources.
        r_wiki: Ratio for Wikipedia lines.
        r_fine_web: Ratio for FineWeb lines.
        r_stories: Ratio for TinyStories lines.
        r_open_web: Ratio for OpenWebText lines.
        seed: Random seed used when shuffling streaming datasets.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_r = r_wiki + r_fine_web + r_stories + r_open_web
    n_wiki = max_lines * r_wiki // total_r
    n_fine_web = max_lines * r_fine_web // total_r
    n_stories = max_lines * r_stories // total_r
    n_open_web = max_lines - n_wiki - n_fine_web - n_stories

    print(f"Wikipedia: {n_wiki:,} lines")
    print(f"FineWeb: {n_fine_web:,} lines")
    print(f"Stories: {n_stories:,} lines")
    print(f"Open Web: {n_open_web:,} lines")
    print("Loading datasets...")

    # Split is not needed for LLMs, take "train" as it has most of the data
    # Use streaming mode to avoid downloading the full dataset

    wiki: IterableDataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    fine_web: IterableDataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    stories: IterableDataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    open_web: IterableDataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    # Shuffle streaming datasets (buffered shuffle)
    wiki = wiki.shuffle(buffer_size=100_000, seed=seed)
    fine_web = fine_web.shuffle(buffer_size=100_000, seed=seed)
    stories = stories.shuffle(buffer_size=10_000, seed=seed)
    open_web = open_web.shuffle(buffer_size=100_000, seed=seed)

    print("Writing corpus...")
    wrote = 0

    with out_path.open("w", encoding="utf-8") as f:
        web_iter = yield_text(fine_web)
        for text in tqdm(take_n(yield_text(wiki), n_wiki), total=n_wiki, desc="Wiki"):
            f.write(text + "\n")
            wrote += 1

        for text in tqdm(take_n(web_iter, n_fine_web), total=n_fine_web, desc="FineWeb"):
            f.write(text + "\n")
            wrote += 1

        for text in tqdm(take_n(yield_text(stories), n_stories), total=n_stories, desc="Stories"):
            f.write(text + "\n")
            wrote += 1

        for text in tqdm(take_n(yield_text(open_web), n_open_web), total=n_open_web, desc="Open Web"):
            f.write(text + "\n")
            wrote += 1

        # Fallback: if any source ran out early, fill the remainder with FineWeb
        missing = max_lines - wrote
        if missing > 0:
            print(f"Filling missing lines with FineWeb: {missing:,}")
            for text in tqdm(take_n(web_iter, missing), total=missing, desc="FineWeb (fill)"):
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
    ap.add_argument("--out", default="artifacts/corpus.txt", help="Output corpus file path.")
    ap.add_argument("--max_lines", type=int, default=10_000_000, help="Max lines to write.")
    ap.add_argument("--ratio-wiki", type=int, default=1, help="Wiki ratio (e.g. 2).")
    ap.add_argument("--ratio-fine-web", type=int, default=2, help="FineWeb ratio (e.g. 2).")
    ap.add_argument("--ratio-stories", type=int, default=1, help="Stories ratio (e.g. 2).")
    ap.add_argument("--ratio-open-web", type=int, default=1, help="OpenWebText ratio (e.g. 2).")
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
        r_fine_web=args.ratio_fine_web,
        r_stories=args.ratio_stories,
        r_open_web=args.ratio_open_web,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
