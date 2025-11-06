# scripts/prepare_dataset.py
import argparse
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def yield_text(ds, field="text"):
    """Extract and clean text from dataset records."""
    for r in ds:
        t = r.get(field)
        if isinstance(t, str):
            t = t.strip().replace("\n", " ")
            # Filter out very short lines (noise)
            if len(t) > 20:
                yield t

def take_n(generator, n):
    """Take first n elements from generator."""
    for i, x in enumerate(generator):
        if i >= n:
            break
        yield x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--max_lines", type=int, default=2_000_000)
    ap.add_argument("--ratio", default="4,4,2", help="wiki,web,stories")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate lines per source
    r_wiki, r_web, r_st = map(int, args.ratio.split(","))
    total_r = r_wiki + r_web + r_st
    n_wiki = args.max_lines * r_wiki // total_r
    n_web  = args.max_lines * r_web  // total_r
    n_st   = args.max_lines - n_wiki - n_web

    print(f"Wikipedia: {n_wiki:,} lines");
    print(f"Web: {n_web:,} lines");
    print(f"Stories: {n_st:,} lines");
    print("Loading datasets...");

    # Load streaming datasets
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1",
                        split="train", streaming=True)
    web  = load_dataset("HuggingFaceFW/fineweb",
                        name="sample-10BT", split="train", streaming=True)
    stories = load_dataset("roneneldan/TinyStories",
                           split="train", streaming=True)
    
    print("Writing corpus...")
    wrote = 0
    
    with open(args.out, "w", encoding="utf-8") as f:
        # Wikipedia (Formal, factual, encyclopedic knowledge)
        for t in tqdm(take_n(yield_text(wiki, "text"), n_wiki), 
                      total=n_wiki, desc="Wiki"):
            f.write(t + "\n")
            wrote += 1

        # Web corpus (Diverse web content, conversational and varied writing styles)
        for t in tqdm(take_n(yield_text(web, "text"), n_web), 
                      total=n_web, desc="Web"):
            f.write(t + "\n")
            wrote += 1

        # Stories (Simple narratives, coherent storytelling)
        for t in tqdm(take_n(yield_text(stories, "text"), n_st), 
                      total=n_st, desc="Stories"):
            f.write(t + "\n")
            wrote += 1
    
    # Final stats
    file_size_gb = Path(args.out).stat().st_size / (1024**3)
    print(f"Saved {wrote:,} lines ({file_size_gb:.2f} GB) -> {args.out}")

if __name__ == "__main__":
    main()