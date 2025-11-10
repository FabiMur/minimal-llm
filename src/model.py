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