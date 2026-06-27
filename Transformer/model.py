import torch
import torch.nn as nn
from typing import Optional
from .config import ModelConfig

class EncoderModel(nn.Module):
    """
    Transformer Encoder Model for sequence representation learning.
    
    This model embeds input sequences, adds learnable positional embeddings,
    processes them through multiple layers of Transformer Encoder, and outputs logits.
    """
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        vocab_size: Optional[int] = None,
        max_len: Optional[int] = None,
        n_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        
        # Determine active settings, supporting both ModelConfig and individual parameter overrides
        if config is None:
            config = ModelConfig()
            
        self.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        self.max_len = max_len if max_len is not None else config.max_len
        self.n_dim = n_dim if n_dim is not None else config.n_dim
        self.n_heads = n_heads if n_heads is not None else config.n_heads
        self.n_layers = n_layers if n_layers is not None else config.n_layers
        self.d_ff = d_ff if d_ff is not None else config.d_ff

        self.embedding = nn.Embedding(self.vocab_size, self.n_dim)
        self.pos_embedding = nn.Embedding(self.max_len, self.n_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_dim,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            batch_first=True
        )

        # Removed 'padding_mask=attention_mask' from here because:
        # 1. It was referring to an undefined variable 'attention_mask' in constructor scope.
        # 2. padding masks are passed during the forward pass in PyTorch's nn.TransformerEncoder.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )

        self.norm = nn.LayerNorm(self.n_dim)
        self.fc = nn.Linear(self.n_dim, self.vocab_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len) where 1 indicates non-padding and 0 indicates padding.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.max_len}")

        # Dynamic position indices creation
        positions = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        # Embeddings combination
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # Convert PyTorch attention mask representation (True means ignore / mask out)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Transformer layers forward pass
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        x = self.norm(x)

        # Linear projection to vocab size
        logits = self.fc(x)

        return logits
