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
