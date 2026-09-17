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

A decoder-only transformer language model built from scratch in PyTorch, inspired by Meta's LLaMA models. Built for learning purposes.

## Demo

The chat TUI (`minimal_llm.chat`, built with [Textual](https://textual.textualize.io/)): a multi-turn ChatML conversation against the SFT-finetuned checkpoint. Generation runs off the UI thread, so the input box disables while the model replies and re-enables the moment it's done.

![minimal-llm chat TUI answering "What is the capital of France?" then "Write a short poem about the ocean.", each reply appearing below the previous turn](tapes/01-chat-demo.gif)

## Architecture

~393M parameter model with the following design:

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

**Default config:** `vocab_size=32000`, `context_length=2048`, `d_model=1024`, `n_layers=32`, `n_heads=16`, `n_kv_heads=4`

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
uv run python -m minimal_llm.data.build_corpus
```

The corpus size targets **2x Chinchilla**. The [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) puts the compute-optimal point at ~20 tokens per parameter, which for this ~393M parameter model is ~7.9B tokens. Meanwhile this pipeline defaults to 20M lines (roughly 15–16B tokens) for ~40 tokens per parameter.

Training past the compute-optimal point is deliberate. Chinchilla asks which split between model size and training tokens minimizes loss for a fixed training budget, and says nothing about inference. The cost of a forward pass depends on model size alone, not on how many tokens the model was trained on: training data is paid for once, parameters are paid for on every token generated. Reaching a given loss with a smaller model and more tokens therefore costs more to train and less to run, same reasoning behind the [LLaMA](https://arxiv.org/abs/2302.13971) models, which are trained well beyond compute-optimal for their size. Not all of the corpus needs to be consumed, `--max_steps` controls how much is actually seen.

### 2. Train the tokenizer

BPE tokenizer with 32K vocab and special tokens `[PAD]`, `[BOS]`, `[EOS]`, plus the ChatML delimiters `<|im_start|>` / `<|im_end|>`. The chat tokens never appear in the pretraining corpus, they are declared up front so the vocab (and therefore the embedding matrix) stays fixed when fine-tuning on chat data later.

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

Omit `--prompt` to drop into an interactive generation loop instead.

### 6. Build the chat (SFT) corpus

Streams and mixes [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) and [`teknium/OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5) (1:10 ratio by default) into a ChatML-ready JSONL corpus:

```bash
uv run python -m minimal_llm.data.sft.build_chat_corpus \
  --out artifacts/chat_corpus.jsonl \
  --max_conversations 110000
```

### 7. Tokenize the chat corpus

Produces `train_ids.bin`, `train_labels.bin`, `val_ids.bin`, `val_labels.bin`, and `meta_chat.json` in `artifacts/`. Loss labels are masked (`-1`) everywhere except assistant turns, so the model is only trained to predict replies, not prompts:

```bash
uv run python -m minimal_llm.data.sft.tokenize_chat \
  --corpus artifacts/chat_corpus.jsonl \
  --tokenizer artifacts/tokenizer.json
```

### 8. Fine-tune (SFT)

Starts from a pretrained checkpoint. The tokenizer already reserves `<|im_start|>`/`<|im_end|>` from step 2, so no embedding resize is needed:

```bash
uv run python -m minimal_llm.sft_train \
  --init_checkpoint artifacts/checkpoints/my_run/best.pt \
  --run_name my_sft_run \
  --max_steps 2000
```

Checkpoints land in `artifacts/checkpoints/<run_name>/`, same layout and `--resume`/`--save_interval` semantics as pretraining.

### 9. Chat

Launches a [Textual](https://textual.textualize.io/) TUI for a multi-turn ChatML conversation against an SFT-finetuned checkpoint (see the [Demo](#demo) above):

```bash
uv run python -m minimal_llm.chat \
  --checkpoint artifacts/checkpoints/my_sft_run/best.pt \
  --tokenizer artifacts/tokenizer.json
```

## Docker

Requires an NVIDIA GPU with CUDA support.

```bash
# Build
docker build -f docker/train/Dockerfile -t minimal-llm-train .
docker build -f docker/infer/Dockerfile -t minimal-llm-infer .

# Pretraining
docker run --gpus all -v $(pwd)/artifacts:/app/artifacts minimal-llm-train \
  --run_name my_run --max_steps 10000 --lr 3e-4

# SFT (same image, override the entrypoint — no separate Dockerfile needed)
docker run --gpus all -v $(pwd)/artifacts:/app/artifacts --entrypoint python minimal-llm-train \
  -m minimal_llm.sft_train --init_checkpoint artifacts/checkpoints/my_run/best.pt --run_name my_sft_run

# Inference
# (Not implemented yet)
```

## Project structure

```
src/minimal_llm/
├── model.py        # Model architecture (TransformerLM, ModelConfig, ...)
├── train.py        # Pretraining loop, optimizer, scheduler, checkpointing
├── sft_train.py    # SFT fine-tuning loop (reuses train.py's building blocks)
├── generate.py     # Inference script
├── chat.py         # Textual TUI for multi-turn ChatML chat
└── data/
    ├── build_corpus.py      # Corpus construction from HuggingFace datasets
    ├── train_tokenizer.py   # BPE tokenizer training
    ├── tokenize_to_bin.py   # Tokenization to binary format
    ├── data_loaders.py      # BinTokenDataset and DataLoader utilities
    └── sft/
        ├── build_chat_corpus.py  # Mixes no_robots + OpenHermes-2.5 into ChatML JSONL
        ├── tokenize_chat.py      # ChatML tokenization with assistant-only loss masking
        └── data_loaders.py       # ChatBinDataset and DataLoader utilities

artifacts/             # Generated files (gitignored)
├── corpus.txt
├── tokenizer.json
├── train.bin / val.bin / meta.json                                  # pretraining data
├── chat_corpus.jsonl
├── train_ids.bin / train_labels.bin / val_ids.bin / val_labels.bin  # SFT data
├── meta_chat.json
└── checkpoints/
    └── <run_name>/
        ├── best.pt
        └── latest.pt
```

## Roadmap

- [ ] Inference Docker image
- [ ] Evaluation (perplexity benchmarks beyond val loss)

## License

MIT
