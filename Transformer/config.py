from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 30000
    max_len: int = 128
    n_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    checkpoint_dir: str = "checkpoints"
    val_split: float = 0.2
    seed: int = 42
    gradient_clip_val: Optional[float] = 1.0
    early_stopping_patience: Optional[int] = 3
