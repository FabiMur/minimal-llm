"""Model definition for a minimal transformer-based language model."""

from dataclasses import dataclass


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
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
