from dataclasses import dataclass
from pathlib import Path
import numpy as np
import random
import torch
import os


@dataclass
class TransformerConfig:
    # Model hyperparameters->determine the architecture of the model
    hidden_dim: int = 512       # size of embedding vectors
    num_heads: int = 8          # number of attention heads
    num_layers: int = 3         # number of encoder and decoder layers
    d_ff: int = 2048            # inner size of the feed-forward block
    dropout_rate: float = 0.1   # dropout probability

    # Training hyperparameters->determine how the model updates itself
    lr: float = 3e-4            # 3e-4=0.0003
    batch_size: int = 8
    num_epochs: int = 20
    weight_decay: float = 1e-4

    # Sequence and language settings
    seq_len: int = 350
    lang_src: str = 'en'        # source language: English
    lang_tgt: str = 'orm'       # target language: Afaan Oromo

    # File paths
    model_folder: str = 'checkpoints'
    model_basename: str = 'tmodel_'
    preload: str = None         # set to an epoch string like '05' to resume training
    tokenizer_file: str = 'tokenizer_{0}.json'
    experiment_name: str = 'runs/tmodel'

    def __post_init__(self):
        """Validate config values after creation."""
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        if self.lr <= 0:
            raise ValueError(f"lr (learning rate) must be positive, got {self.lr}")


def get_weight_file_path(config: TransformerConfig, epoch: str) -> str:
    """Return the full file path for a model checkpoint at a given epoch."""
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path('.') / config.model_folder / model_filename)


def set_seed(seed: int = 42) -> None:
    """Set random seeds everywhere so experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)