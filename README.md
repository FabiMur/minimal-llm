# minimal-llm

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/badge/uv-package%20manager-7c3aed.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)
[![pyright](https://img.shields.io/badge/type%20checked-pyright-1674b1.svg)](https://github.com/microsoft/pyright)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0a9edc.svg)](https://pytest.org)

A decoder-only transformer language model built from scratch in PyTorch, inspired by Meta's LLaMA models. Built for learning purposes.

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
- GPT-2 style init: all linear and embedding weights drawn from `N(0, 0.02)` — [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- SwiGLU hidden dim: `4 * d_model * 2/3` rounded up to the nearest multiple of 256 (2816 at `d_model=1024`) — [Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- RMSNorm pre-normalization — [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
- AdamW with `β=(0.9, 0.95)` and cosine LR schedule with linear warmup — [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)
- RoPE positional encoding with split-half formulation — [Su et al., 2023](https://arxiv.org/abs/2104.09864)
- Grouped Query Attention (GQA) with `n_kv_heads=4`: 4 KV heads shared across 16 Q heads, reducing KV cache size 4x at inference — [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)
- KV Cache for inference: pre-allocated per-layer buffers that are prefilled once over the prompt, then each decode step processes only the new tokens query and appends new K/V pairs to the cache — [LLaMA, 2023](https://arxiv.org/abs/2302.13971)
- Causal masking via `F.scaled_dot_product_attention(is_causal=True)`, which dispatches to Flash Attention when available
- Gradient checkpointing on every transformer block (enabled by default during training, disabled when a KV cache is active)

**Primary references:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla) — Hoffmann et al., 2022

**Default config** (as exposed by `train.py`): `vocab_size=32000`, `context_length=2048`, `d_model=1024`, `n_layers=32`, `n_heads=16`, `n_kv_heads=4`, `rope_theta=10000.0`

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/FabiMur/minimal-llm.git
cd minimal-llm
uv sync
```

## Usage

### 1. Build the corpus

Streams Wikipedia, FineWeb, TinyStories, and OpenWebText into a single text file, one document per line, mixed with a default ratio of `1 : 2 : 1 : 1`:

```bash
uv run python -m minimal_llm.data.build_corpus
```

Each source is shuffled with a buffered streaming shuffle; if a source runs out of data early, the shortfall is filled with FineWeb. Ratios and size are configurable (`--ratio-wiki`, `--ratio-fine-web`, `--ratio-stories`, `--ratio-open-web`, `--max_lines`).

The corpus size targets **2x Chinchilla**. The [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) puts the compute-optimal point at ~20 tokens per parameter, which for this ~393M parameter model is ~7.9B tokens. Meanwhile this pipeline defaults to 23M lines (roughly 18B tokens) for ~45 tokens per parameter.

Training past the compute-optimal point is deliberate. Chinchilla asks which split between model size and training tokens minimizes loss for a fixed training budget, and says nothing about inference. The cost of a forward pass depends on model size alone, not on how many tokens the model was trained on: training data is paid for once, parameters are paid for on every token generated. Reaching a given loss with a smaller model and more tokens therefore costs more to train and less to run, same reasoning behind the [LLaMA](https://arxiv.org/abs/2302.13971) models, which are trained well beyond compute-optimal for their size. Not all of the corpus needs to be consumed, `--max_steps` controls how much is actually seen.

### 2. Train the tokenizer

Byte-level BPE tokenizer with 32K vocab and special tokens `<|pad|>`, `<|bos|>`, `<|eos|>`, plus the ChatML delimiters `<|im_start|>` / `<|im_end|>`. The chat tokens never appear in the pretraining corpus, they are declared up front so the vocab (and therefore the embedding matrix) stays fixed when fine-tuning on chat data later.

```bash
uv run python -m minimal_llm.data.train_tokenizer \
  --corpus artifacts/corpus.txt \
  --vocab_size 32000
```

### 3. Tokenize to binary

Produces `train.bin`, `val.bin`, and `meta.json` in `artifacts/`. Each line is encoded and wrapped in `<|bos|>` / `<|eos|>`, then routed to the train or validation split (1% by default) and appended as flat `uint16` token IDs:

```bash
uv run python -m minimal_llm.data.tokenize_to_bin \
  --corpus artifacts/corpus.txt \
  --tokenizer artifacts/tokenizer.json
```

`meta.json` records the dtype, the split token counts, and the paths of both binaries, which is all `train.py` needs to build its dataloaders. Training reads the binaries through `np.memmap`, so the corpus never has to fit in RAM.

### 4. Train

```bash
uv run python -m minimal_llm.train \
  --run_name my_run \
  --max_steps 10000 \
  --lr 3e-4 \
  --warmup_steps 500
```

Defaults are tuned for a single A100: `batch_size=16` with `grad_accum_steps=16` gives an effective batch of 256 sequences (~524K tokens per optimizer step), and the default `max_steps=34000` consumes ~17.8B tokens. Training runs under bfloat16 autocast on CUDA (plain float32 on MPS/CPU), with `torch.compile` and gradient checkpointing both on by default — disable them with `--no_compile` and `--no_grad_checkpoint`.

Checkpoints are saved to `artifacts/checkpoints/<run_name>/`: `latest.pt` every `--save_interval` steps and `best.pt` whenever validation loss improves. Both store the model, optimizer, scheduler, step, and the run's arguments, so a run can be resumed with `--resume artifacts/checkpoints/my_run/latest.pt`.

### 5. Generate

```bash
uv run python -m minimal_llm.generate \
  --checkpoint artifacts/checkpoints/my_run/best.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Once upon a time"
```

The model config is rebuilt from the arguments stored in the checkpoint, so no model flags are needed. Sampling is controlled with `--num_new_tokens`, `--temperature`, and `--top_k`; `<|bos|>` is prepended to the prompt unless `--no_bos` is passed. Omitting `--prompt` starts an interactive loop (empty line to quit). Generation uses the KV cache: one prefill pass over the prompt, then one token per decode step.

## Docker

Requires an NVIDIA GPU with CUDA support.

```bash
# Build
docker build -f docker/train/Dockerfile -t minimal-llm-train .

# Training (arguments are forwarded to minimal_llm.train)
docker run --gpus all -v $(pwd)/artifacts:/app/artifacts minimal-llm-train \
  --run_name my_run --max_steps 10000 --lr 3e-4

# Inference
# (docker/infer/Dockerfile is still empty — not implemented yet)
```

## Development

Linting, type checking, and tests run through the dev dependency group:

```bash
uv sync --all-groups      # install dev tools
uv run ruff check .       # lint (line length 120, pydocstyle google convention)
uv run ruff format .      # format
uv run pyright            # type check
uv run pytest             # tests
```

The same three checks run as [pre-commit](https://pre-commit.com/) hooks:

```bash
uv run pre-commit install
```

## Project structure

```
src/minimal_llm/
├── model.py          # Model architecture (TransformerLM, ModelConfig, KVCache, ...)
├── train.py          # Training loop, optimizer, scheduler, checkpointing
├── generate.py       # Inference script (checkpoint loading, sampling, interactive mode)
└── data/
    ├── build_corpus.py      # Corpus construction from HuggingFace datasets
    ├── train_tokenizer.py   # BPE tokenizer training
    ├── tokenize_to_bin.py   # Tokenization to binary format
    └── data_loaders.py      # BinTokenDataset and DataLoader utilities

docker/
├── train/Dockerfile  # CUDA training image
└── infer/Dockerfile  # Inference image (empty, pending)

tests/                # pytest suite

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

- [x] Inference script (`generate.py`)
- [ ] Inference Docker image
- [ ] Evaluation (perplexity benchmarks beyond val loss)
- [ ] Testing (only a placeholder smoke test so far)
- [ ] Chat fine-tuning on ChatML-formatted data

## License

MIT
