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
    - Dropout: applied after the final projection for regularization

    This variant increases performance whithout a significant increase in parameters.
    """ 
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Standard practice: hidden dim is 4x the model dimension
 
        # Reduced hidden_dim (like LLaMA): 4 * d_model * (2/3)
        hidden_dim = int(4 * config.d_model * 2/3)

        # SwiGLU components
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=False)  # gate
        self.fc2 = nn.Linear(config.d_model, hidden_dim, bias=False)  # up-projection
        self.fc3 = nn.Linear(hidden_dim, config.d_model, bias=False)  # down-projection

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """Forward pass of feed-forward network.
            Input: tensor of shape (batch_size, seq_len, d_model)
            Output: tensor of shape (batch_size, seq_len, d_model)
        """
        x = F.silu(self.fc1(x)) * self.fc2(x)  # SwiGLU
        x = self.fc3(x)
        x = self.dropout(x)
        
        return x
    
class TransformerBlock(nn.Module):
    """A single transformer decoder block.
    
    This combines attention and feed-forward with layer normalization and residual connections.
    The structure follows the modern Pre-LN version, which is more stable during training:
    
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    
    The residual connections (the + x part) help gradients flow during backpropagation.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x

class GPTModel(nn.Module):
    """GPT Language Model.

    The complete model that combines all components:
    - Token embeddings
    - Position embeddings
    - Transformer blocks
    - Output projection
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings: vector representation for each token in vocabulary
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embeddings: vector representation for each position in the sequence
        # Note: This is crucial because transformers have no inherent notion of order
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)

        # Token embedding and position embedding values are learned during training
        # Token embedding and position embedding are summed to form the input to the transformer blocks

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Normalization layer before output
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection to vocabulary
        # This maps the final hidden states back to logits over the vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and often improves performance
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights with the chosen strategy
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization scheme.
        
        Linear layers and embeddings are initialized from a normal distribution with mean 0 and std 0.02.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """Forward pass through the model.
        
        Args:
            idx: Matrix of input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs for computing loss, same shape as idx
            
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: tuple of (logits, loss)
        """
        batch_size, seq_len = idx.shape
        
        # Transform token IDs to token embeddings
        # Shape: (batch_size, seq_len, d_model)
        token_emb = self.token_embedding(idx)
        
        # Get position embeddings for positions 0, 1, 2, ..., seq_len-1
        # Shape: (seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Add token and position embeddings
        # Broadcasting handles adding pos_emb to each sequence in the batch
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary to get logits
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # Optionally compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross entropy: it expects (N, C) where N is batch*seq_len
            # and C is number of classes (vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (batch_size * seq_len, vocab_size)
                targets.view(-1),                   # (batch_size * seq_len)
                ignore_index=-1                     # Ignore padding tokens
            )
        
        return logits, loss
