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
    rope_theta: float = 10000.0  # Base frequency for RoPE

    def __post_init__(self):
        """Validate that d_model is divisible by n_heads.

        Each attention head will manage d_model // n_heads dimensions.
        """
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


def precompute_rope_freqs(
    head_dim: int,
    context_length: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency tensors for RoPE.

    Uses the "split half" pairing formulation of RoPE, which is equivalent to the original
    interleaved version but more memory-efficient. This follows the LLaMA implementation and
    is the most common variant used in modern transformer models.

    Args:
        head_dim: Dimension of each attention head. Must be even.
        context_length: Maximum sequence length to precompute frequencies for.
        theta: Base frequency. Defaults to 10000.0 as in the original RoPE paper.
        device: Device to create tensors on.

    Returns:
        Tuple of (cos, sin) tensors of shape (context_length, head_dim // 2).
    """
    # Frequency for each pair of dimensions: theta^(-2i/head_dim) for i in [0, head_dim/2)
    # head_dim is used instead of d_model because RoPE is applied separately to each attention head
    # i.e., a given dimension pair will have the same frequency across all heads,
    # but different pairs will have different frequencies.
    # Shape: (head_dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices: 0, 1, 2, ..., context_length - 1
    # Shape: (context_length,)
    positions = torch.arange(context_length, device=device)

    # Outer product: angle for each (position, frequency) pair
    # Shape: (context_length, head_dim // 2)
    angles = torch.outer(positions, freqs)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings to a query or key tensor.

    Splits the head dimension in half and applies a 2D rotation to each pair:
        [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]

    Args:
        x: Tensor of shape (batch_size, n_heads, seq_len, head_dim).
        cos: Cosine frequencies of shape (seq_len, head_dim // 2).
        sin: Sine frequencies of shape (seq_len, head_dim // 2).

    Returns:
        Rotated tensor of the same shape as x.
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]  # (batch, n_heads, seq_len, head_dim // 2)
    x2 = x[..., head_dim // 2 :]  # (batch, n_heads, seq_len, head_dim // 2)

    # Broadcast cos/sin over batch and head dimensions to match x's shape for elementwise operations
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply the rotation to each pair of dimensions and concatenate back together
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            cos: RoPE cosine frequencies of shape (seq_len, head_dim // 2)
            sin: RoPE sine frequencies of shape (seq_len, head_dim // 2)

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
        q, k, v = qkv.unbind(0)

        # Apply RoPE to queries and keys (not values — position only affects matching, not content)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Compute scaled dot-product attention with causal masking
        # This efficiently computes: softmax(QK^T / sqrt(d_k)) V
        # The is_causal=True flag ensures tokens only attend to previous
        # positions by applying a mask to the attention scores.
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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

    This variant increases performance without a significant increase in parameters.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network.

        Input: tensor of shape (batch_size, seq_len, d_model)
        Output: tensor of shape (batch_size, seq_len, d_model)
        """
        x = F.silu(self.fc1(x)) * self.fc2(x)  # SwiGLU
        x = self.fc3(x)

        return x


class TransformerBlock(nn.Module):
    """A single transformer decoder block.

    This combines attention and feed-forward with layer normalization and residual connections.
    The structure follows the modern Pre-LN version, which is more stable during training:

    x = x + Attention(RMSNorm(x))
    x = x + FeedForward(RMSNorm(x))

    The residual connections (the + x part) help gradients flow during backpropagation.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the transformer block."""
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            cos: RoPE cosine frequencies of shape (seq_len, head_dim // 2)
            sin: RoPE sine frequencies of shape (seq_len, head_dim // 2)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x), cos, sin)

        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    """Transformer Language Model (Decoder-only architecture).

    The complete model that combines all components:
    - Token embeddings
    - Transformer blocks
    - Output projection
    """

    def __init__(self, config: ModelConfig):
        """Initialize the Transformer Language Model."""
        super().__init__()
        self.config = config

        # Token embeddings: vector representation for each token in vocabulary
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Normalization layer before output
        self.ln_f = nn.RMSNorm(config.d_model)

        # Output projection to vocabulary
        # This maps the final hidden states back to logits over the vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights with the chosen strategy
        self.apply(self._init_weights)

        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and often improves performance
        self.lm_head.weight = self.token_embedding.weight

        # Precompute RoPE frequencies and register as non-trainable buffers
        # Buffers are moved to the correct device automatically with .to(device)
        rope_cos, rope_sin = precompute_rope_freqs(
            head_dim=config.d_model // config.n_heads,
            context_length=config.context_length,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        self.rope_cos: torch.Tensor
        self.rope_sin: torch.Tensor

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 initialization scheme.

        Linear layers and embeddings are initialized from a normal distribution with mean 0 and std 0.02.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the model.

        Args:
            idx: Matrix of input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs for computing loss, same shape as idx

        Returns:
            logits: Unnormalized scores for each token in the vocabulary, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets are provided, else None
        """
        batch_size, seq_len = idx.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model's context length ({self.config.context_length})"
            )

        # Transform token IDs to token embeddings
        # Shape: (batch_size, seq_len, d_model)
        x = self.token_embedding(idx)

        # Slice precomputed RoPE frequencies for the current sequence length
        cos = self.rope_cos[:seq_len]  # (seq_len, head_dim // 2)
        sin = self.rope_sin[:seq_len]  # (seq_len, head_dim // 2)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

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
                logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
                targets.view(-1),  # (batch_size * seq_len)
                ignore_index=-1,  # Ignore padding tokens, marked with -1 in targets
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self, idx: torch.Tensor, num_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Generate new tokens from the model given an initial prompt (idx).

        This method is used for inference: given a prompt (idx), generate new tokens
        one at a time by repeatedly sampling from the model's output distribution.

        Args:
            idx: Input matrix of token IDs of shape (batch_size, seq_len)
            num_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            Generated sequence of shape (batch_size, seq_len + num_new_tokens)
        """
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        for _ in range(num_new_tokens):
            # Crop context if it exceeds maximum context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length :]

            # Get predictions for next token
            logits, _ = self(idx_cond)

            # Focus only on the last position (next token prediction) and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering if specified
            if top_k is not None:
                # Mask out all tokens that are not in the top k threshold
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
