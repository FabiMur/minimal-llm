# minimal-llm

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/badge/uv-package%20manager-7c3aed.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)
[![pyright](https://img.shields.io/badge/type%20checked-pyright-1674b1.svg)](https://github.com/microsoft/pyright)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0a9edc.svg)](https://pytest.org)

A decoder-only transformer language model built from scratch in PyTorch, inspired by Meta's LLaMA models. Built for learning pourpuses.

## Architecture

~239M parameter model with the following design:

| Component | Choice |
|---|---|
| Architecture | Decoder-only transformer (causal LM) |
| Normalization | Pre-LN with RMSNorm |
| Feed-forward | SwiGLU (`SiLU(gate) * value`) |
| Position encoding | RoPE (Rotary Position Embeddings) |
| Attention | Grouped Query Attention (GQA) with `F.scaled_dot_product_attention` |
| Precision | bfloat16 (Ampere+ GPUs) |

**Key design choices:**
- No biases in any linear layer
- Weight tying between `token_embedding` and `lm_head` — [Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)
- GPT-2 style init: `N(0, 0.02)`, residual projections scaled by `1/sqrt(2 * n_layers)` — [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- SwiGLU hidden dim: `⌈4 * d_model * 2/3⌉` rounded to nearest multiple of 256 — [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- RMSNorm pre-normalization — [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
- AdamW with `β=(0.9, 0.95)` and cosine LR schedule with linear warmup — [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)
- RoPE positional encoding with split-half formulation — [Su et al., 2023](https://arxiv.org/abs/2104.09864)
- Grouped Query Attention (GQA) with `n_kv_heads=4`: 4 KV heads shared across 16 Q heads, reducing KV cache size 4x at inference — [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)
- KV Cache for inference: pre-allocated per-layer buffers that are prefilled once over the prompt, then each decode step processes only the new tokens query and appends new K/V pairs to the cache — [LLaMA, 2023](https://arxiv.org/abs/2302.13971)
- Causal masking via `F.scaled_dot_product_attention(is_causal=True)`, which dispatches to Flash Attention when available

**Primary references:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla) — Hoffmann et al., 2022

**Default config:** `vocab_size=32000`, `context_length=1024`, `d_model=1024`, `n_layers=16`, `n_heads=16`, `n_kv_heads=4`

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/FabiMur/minimal-llm.git
cd minimal-llm
uv sync
```

## Usage

### 1. Build the corpus

Streams Wikipedia, FineWeb, TinyStories, and OpenWebText into a single text file:

```bash
uv run python -m minimal_llm.data.build_corpus --max_lines 10000000
```

The corpus size follows the [Chinchilla scaling law](https://arxiv.org/abs/2203.15556), which recommends ~20 tokens per parameter for compute-optimal training. For a ~239M parameter model that means ~4.8B tokens. The pipeline downloads 6–7B tokens to have headroom — not all of them need to be used.

### 2. Train the tokenizer

BPE tokenizer with 32K vocab and special tokens `[PAD]`, `[BOS]`, `[EOS]`:

```bash
uv run python -m minimal_llm.data.train_tokenizer \
  --corpus artifacts/corpus.txt \
  --vocab_size 32000
```

### 3. Tokenize to binary

Produces `train.bin`, `val.bin`, and `meta.json` in `artifacts/`:

```bash
uv run python -m minimal_llm.data.tokenize_to_bin \
  --corpus artifacts/corpus.txt \
  --tokenizer artifacts/tokenizer.json
```

### 4. Train

```bash
uv run python -m minimal_llm.train \
  --run_name my_run \
  --max_steps 10000 \
  --lr 3e-4 \
  --warmup_steps 500
```

Checkpoints are saved to `artifacts/checkpoints/<run_name>/`. The best validation loss checkpoint is saved as `best.pt`.

### 5. Generate

```bash
uv run python -m minimal_llm.generate \
  --checkpoint artifacts/checkpoints/my_run/best.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Once upon a time"
```

## Docker

Requires an NVIDIA GPU with CUDA support.

```bash
# Build
docker build -f docker/train/Dockerfile -t minimal-llm-train .
docker build -f docker/infer/Dockerfile -t minimal-llm-infer .

# Trainining
docker run --gpus all -v $(pwd)/artifacts:/app/artifacts minimal-llm-train \
  --run_name my_run --max_steps 10000 --lr 3e-4

# Inference
# (Not implemented yet)
```

## Project structure

```
src/minimal_llm/
├── model.py          # Model architecture (TransformerLM, ModelConfig, ...)
├── train.py          # Training loop, optimizer, scheduler, checkpointing
├── generate.py       # Inference script
└── data/
    ├── build_corpus.py      # Corpus construction from HuggingFace datasets
    ├── train_tokenizer.py   # BPE tokenizer training
    ├── tokenize_to_bin.py   # Tokenization to binary format
    └── data_loaders.py      # BinTokenDataset and DataLoader utilities

artifacts/            # Generated files (gitignored)
├── corpus.txt
├── tokenizer.json
├── train.bin
├── val.bin
├── meta.json
└── checkpoints/
    └── <run_name>/
        ├── best.pt
        └── latest.pt
```

## Roadmap

- [ ] Inference script (`generate.py`)
- [ ] Inference Docker image
- [ ] Evaluation (perplexity benchmarks beyond val loss)
- [ ] Testing

## License

MIT
