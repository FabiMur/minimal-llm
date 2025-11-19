# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for GPT model architecture.

    Holds all the hyperparameters needed to build the GPT model.
    """
    vocab_size: int = 16000      # Size of the vocabulary (number of unique tokens)
    context_length: int = 1024   # Maximum sequence length the model can handle
    d_model: int = 1024          # Dimensionality of token embeddings and model hidden states
    n_layers: int = 16           # Number of transformer blocks to stack
    n_heads: int = 16            # Number of attention heads per block
    dropout: float = 0.1         # Dropout probability for regularization (attention dropout and residual dropout)
    
    def __post_init__(self):
        """Validate that d_model is divisible by n_heads.
        
        Each attention head will get d_model // n_heads dimensions.
        If this doesn't divide evenly, we can't split the dimensions across heads.
        """
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This is the core of the transformer. It allows each token to gather information from
    all previous tokens in the sequence. Multiple heads can learn to focus on different aspects
    of the input.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # These 3 projections transform the input into queries, keys, and values
        # They are combined into a single linear layer for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        
        # Final output projection to combine all head outputs
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Dropout for regularization
        # It's usual to apply the same dropout rate to attention weights and output
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Create causal mask: a lower triangular matrix
        # This ensures token i can only attend to tokens 0...i (not future tokens)
        # The view function transforms the context_length x context_length causal matrix into
        # a tensor. In this case a 1 x 1 matrix of context_length x context_length matrices.
        # This allows pytorch to broadcast(repeat the values) the mask across batches and heads 
        # during attention computation.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
                .view(1, 1, config.context_length, config.context_length)
        )


    
    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project input to queries, keys, and values
        # Shape: (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)
        
        # Split QKV into another 2 dimensions: one for the 3 (Q, K, V) and one for the heads
        # Shape: (batch_size, seq_len, 3, n_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)

        # Permute dimensions to get the order we want
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)

        # Split into Q, K, V
        # Shape of each: (batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # The scaling factor prevents the dot products from growing too large
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        # Original formula from "Attention is All You Need" paper
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        # The V multiplication is done later after softmax
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask: set future positions to -inf so they get 0 probability after softmax
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # Convert scores to probabilities
        attn_probs = F.softmax(attn_scores, dim=-1) 
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        out = attn_probs @ v
        
        # Concatenate heads and project to output
        # Shape: (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out
class FeedForward(nn.Module):
    """Position-wise feed-forward network.
    
    This is applied independently to each position. It's a simple two-layer MLP
    with a GELU activation in between. This component allows the model to process
    the information gathered by attention, applying non-linear transformations 
    that can capture complex patterns.

    - Layer 1: Linear transformation from d_model to 4 * d_model
    - Activation: GELU (Gaussian Error Linear Unit)
    - Layer 2: Linear transformation back to d_model
    - Dropout: Applied after the second linear layer to prevent overfitting

    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Standard practice: hidden dim is 4x the model dimension
        hidden_dim = 4 * config.d_model
        
        self.fc1 = nn.Linear(config.d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """Forward pass of feed-forward network.
            Input: tensor of shape (batch_size, seq_len, d_model)
            Output: tensor of shape (batch_size, seq_len, d_model)
        """
        # Expand to hidden dimension with GELU activation
        x = self.fc1(x)

        # Apply GELU activation (works better than ReLU for language models)
        x = self.act(x)
        
        # Project back to model dimension
        x = self.fc2(x)

        # Apply dropout for regularization
        x = self.dropout(x)
        
        return x