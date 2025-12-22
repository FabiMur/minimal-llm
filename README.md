# 🐜 Minimal LLM - An LLM from Scratch

![Project Cover](./assets/cover.png)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

## 📋 Table of Contents
- [🎯 Overview](#overview)
- [✨ Features](#features)
- [🏗️ Architecture](#architecture)
- [📊 Dataset](#dataset)
- [🛠️ Installation](#installation)
- [📝 Data Preprocessing](#data-preprocessing)
- [⚙️ Model Configuration](#model-configuration)
- [🚂 Training](#training)
- [🌐 Deployment](#deployment)
- [📄 License](#license)


## 🎯 Overview

**Minimal LLM** is a decoder-only Transformer language model implemented fully from scratch in PyTorch. The project focuses on leveraging modern techniques in NLP and deep learning to create a compact yet powerful language model.

The repository covers the full training stack: custom dataset preparation, BPE tokenization, binary token storage with memory-mapped loading, and a scalable Transformer implementation designed to train efficiently on consumer-grade GPUs.

> ⚠️ **Status**: data pipeline and model architecture completed. Training, evaluation, and deployment are in progress.

### Key Objectives
- **Understand the full LLM lifecycle**: implement and connect every stage of a language model pipeline, from data preprocessing to autoregressive generation.

- **Apply modern Transformer techniques**: use contemporary architectural choices to achieve competitive performance with older and significantly larger models such as GPT-2.

- **Practice production-grade PyTorch**: write clean, modular, and well-documented PyTorch code that mirrors real-world research and industry implementations.

- **Train on consumer-grade hardware**: optimize the model and training setup to run efficiently on single-GPU systems.

- **Containerize the workflow**: create Docker images for both training and inference, enabling reproducible experiments and simplified deployment.

- **Prepare for deployment and inference**: design the model and codebase with future deployment in mind, including text generation APIs and lightweight inference containers.

- **Demonstrate end-to-end ownership**: showcase the ability to design, implement, and reason about a complete LLM system without relying on high-level frameworks.

## ✨ Features

- **End-to-end LLM pipeline**: covers dataset construction, BPE tokenizer training, binary token storage, efficient dataloading, model definition, and autoregressive generation.

- **Modern GPT-style architecture**: decoder-only Transformer with Pre-LayerNorm, SwiGLU feed-forward networks, causal self-attention, and weight tying.

- **Efficient data pipeline**: tokenized datasets stored in binary format and loaded via memory-mapped files for fast, low-overhead training on large corpora.

- **Designed for scalable training**: supports long-context training, configurable stride-based sampling, and single-GPU setups.

- **Clean, modular PyTorch implementation**: minimal abstractions, readable code, and explicit implementations of all core components (attention, FFN, blocks).

- **Reproducible experiments**: deterministic dataset splits, explicit configuration objects, and environment definitions via Conda and pre-commit tooling.

- **Inference-ready generation**: built-in autoregressive text generation with temperature and top-k sampling.

- **Deployment-oriented structure**: codebase designed to support future Dockerized training and inference workflows.

## 🏗️ Architecture

The model is a **decoder-only Transformer (GPT-style)** implemented directly in PyTorch, using several modern architectural choices inspired by recent LLMs.

```text
Model Configuration:
- Parameters:        ~220M
- Hidden size:       1024
- Transformer layers: 16
- Attention heads:   16
- Head dimension:    64
- Vocabulary size:   16,000 (Trained from scratch through BPE)
- Context length:    1024 tokens
- Feed-forward:      SwiGLU (LLaMA-style)
- Normalization:     Pre-LN (LayerNorm before attention / MLP)
- Position encoding: Learned absolute embeddings
```
> Parameter count corresponds to the current default configuration (d_model=1024, n_layers=16).

### Model Components
- **Token Embeddings**: Learned embeddings mapping token IDs to dense vectors of size d_model.
- **Position Embeddings**: Learned absolute position embeddings added to token embeddings.

- **Transformer Blocks (×16)**: Each block follows a modern Pre-LN structure:

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```
- **Multi-Head Self-Attention**: 
  - Single QKV projection
  - Causal masking (autoregressive)
  - Scaled dot-product attention
  - Dropout on attention weights and residuals

- **Feed-Forward Network (SwiGLU)**:
  - Hidden dimension: 4 * d_model * 2/3
  - Gated activation: SiLU(W1x) ⊙ W2x
  - Improves parameter efficiency vs. GELU MLPs

- **Final LayerNorm + Output Projection**:
  - Projection back to vocabulary size
  - Weight tying between token embeddings and output head

## 📊 Dataset

## Data Sources

The training corpus is built from a **manually curated mix of heterogeneous text sources** to balance factual knowledge, general web language, and narrative coherence.

The dataset is created via a streaming pipeline using Hugging Face Datasets, with shuffling and noise filtering applied before tokenization.

- **Wikipedia (Wikitext-103)**  
  Formal, factual, encyclopedic text.  
  Used to provide structured knowledge and neutral writing style.

- **FineWeb (sample-10BT)**  
  Large-scale web crawl data.  
  Provides diverse, conversational, and informal language patterns.

- **TinyStories**  
  Short, coherent narrative texts.  
  Helps the model learn basic syntax, storytelling structure, and long-range coherence.

## Dataset Construction
- Sources are mixed using a configurable ratio (default: `4:4:2` → Wikipedia : Web : Stories)
- Streaming loading (no full dataset download)
- Shuffled with fixed seed for reproducibility
- Lines shorter than a minimum length are filtered as noise
- One document per line
- Split into train/validation at tokenization time

## Final Corpus Statistics

> Note: the corpus is built by sampling up to `--max_lines` per-source (ratio-based). In practice, the final number of lines can be lower if a source runs out of samples in streaming mode.

- **Target lines**: 10,000,000 (ratio-based across sources)
- **Final lines**: ~5.5 million *(limited by dataset availability for the requested ratio and target lines)*
- **Total tokens**: ~3.6 billion tokens *(token count is the primary target/metric)*
- **Vocabulary size**: 16,000 (BPE trained from scratch on the final corpus)
- **On-disk artifacts**:
  - `corpus.txt`: ~13.6 GB (raw text)
  - `train.bin`: ~6.9 GB (token IDs, `uint16`)
  - `val.bin`: ~70 MB

The corpus is serialized into flat binary token streams (`train.bin`, `val.bin`) for efficient **memory-mapped** training via NumPy `memmap`.



## 🛠️ Installation

### Prerequisites
- **Conda** (Miniconda or Anaconda)
- Python 3.11 (managed by Conda)
- CUDA-capable GPU (Linux) *or* Apple Silicon GPU (Metal)

> CUDA and Metal dependencies are handled automatically by the environment setup script.

### Setup

```bash
# Clone the repository
git clone https://github.com/FabiMur/minimal-llm.git
cd minimal-llm

# Create / update environment setup script (CUDA or Metal auto-detected)
./setup_env.sh

# Activate the virtual environment
conda activate minimal-llm
```

### Dependencies
Dependencies are managed via Conda and provided for both CUDA (Linux) and Metal (macOS Apple Silicon):

- [environment.cuda.yaml](environment.cuda.yaml)
- [environment.metal.yaml](environment.metal.yaml)

## 📝 Data Preprocessing

## ⚙️ Model Configuration

## 🚂 Training
> 🚧 Training pipeline and experiments will be documented here once completed.

## 🌐 Deployment
> 🚧 Dockerized training and inference images are planned.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.