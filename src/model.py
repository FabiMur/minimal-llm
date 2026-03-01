"""Model definition for a minimal transformer-based language model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    """Configuration for the model architecture.

    Holds all the hyperparameters needed to build the model.
    """

    vocab_size: int = 32000  # Size of the vocabulary (number of unique tokens)
    context_length: int = 1024  # Maximum sequence length the model can handle
    d_model: int = 1024  # Dimensionality of token embeddings and model hidden states
    n_layers: int = 16  # Number of transformer blocks to stack
    n_heads: int = 16  # Number of attention heads per block

    def __post_init__(self):
        """Validate that d_model is divisible by n_heads.

        Each attention head will manage d_model // n_heads dimensions.
        """
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    The core of the transformer. It allows each token to gather information from all previous tokens in the sequence.
    Multiple heads allow the model to focus and learn on different aspects of the input.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the multi-head attention module."""
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # These 3 projection matrices transform the input into queries, keys, and values
        # They are combined into a single matrix for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Final output projection to combine all head outputs
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Project input to queries, keys, and values
        # Input shape: (batch_size, seq_len, d_model)
        # Output shape: (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split QKV into another 2 dimensions: one for the 3 (Q, K, V) and one for the heads
        # Shape: (batch_size, seq_len, 3, n_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)

        # Permute dimensions to be able to split into Q, K, V, batch and heads easily
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)

        # Split into Q, K, V
        # Shape of each: (batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention with causal masking
        # This efficiently computes: softmax(QK^T / sqrt(d_k)) V
        # The is_causal=True flag ensures tokens only attend to previous
        # positions by applying a mask to the attention scores.
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # Concatenate heads, and project to output
        # Shape: (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (SwiGLU variant).

    This module is applied independently to each position. Instead of the classic
    two-layer MLP with GELU (as in the original Transformer), this version uses
    a gated SwiGLU feed-forward network, similar to modern architectures like LLaMA.

    Architecture:
    - Up-projection: two parallel Linear layers expand d_model -> hidden_dim
        * fc1 produces the "gate" vector (passed through SiLU)
        * fc2 produces the "value" vector
    - SwiGLU activation: SiLU(gate) * value
        (gates the information in a learned, elementwise manner)
    - Down-projection: Linear transformation hidden_dim -> d_model

    This variant increases performance whithout a significant increase in parameters.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the feed-forward network."""
        super().__init__()
        # Standard practice: hidden dim is 4x the model dimension

        # Reduced hidden_dim (like LLaMA): 4 * d_model * (2/3)
        hidden_dim = int(4 * config.d_model * 2 / 3)
        # Round up to the nearest multiple of 256 for better GPU efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        # SwiGLU components
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=False)  # gate
        self.fc2 = nn.Linear(config.d_model, hidden_dim, bias=False)  # up-projection
        self.fc3 = nn.Linear(hidden_dim, config.d_model, bias=False)  # down-projection

    def forward(self, x):
        """Forward pass of feed-forward network.

        Input: tensor of shape (batch_size, seq_len, d_model)
        Output: tensor of shape (batch_size, seq_len, d_model)
        """
        x = F.silu(self.fc1(x)) * self.fc2(x)  # SwiGLU
        x = self.fc3(x)

        return x
