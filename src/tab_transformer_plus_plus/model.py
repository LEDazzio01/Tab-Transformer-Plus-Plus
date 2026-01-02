"""Model definitions for TabTransformer++.

The implementation here extracts the core components from the research
notebook and packages them into reusable PyTorch modules.  The model
supports dual token–scalar representations, gated fusion, per‑feature
MLPs, TokenDrop regularisation and a Pre‑LayerNorm Transformer encoder.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn


@dataclass
class TTConfig:
    """Configuration for TabTransformerGated.

    This dataclass collects hyperparameters with sensible defaults.  When
    instantiating the model you can override any of these values.
    """

    emb_dim: int = 64  # Embedding dimension (d_model)
    n_heads: int = 4   # Multi‑head attention heads
    n_layers: int = 3  # Transformer encoder layers
    ffn_multiplier: int = 4  # Feed‑forward expansion factor
    dropout: float = 0.1  # Dropout for attention and FFN
    emb_dropout: float = 0.05  # Post‑embedding dropout
    tokendrop_p: float = 0.12  # TokenDrop probability


class PerTokenValMLP(nn.Module):
    """Per‑token MLP used to project a scalar value into the embedding space.

    Each feature can have its own instance of this module.  The design
    mirrors the notebook: a two‑layer MLP with GELU activation and
    LayerNorm.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenDrop(nn.Module):
    """Randomly drops feature embeddings during training.

    TokenDrop zeroes entire feature embeddings (per sample, per feature)
    with a probability `p` and rescales the remaining embeddings by
    `1/(1-p)` to maintain expected magnitude.  The class always keeps
    the first token (e.g. CLS token) intact.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        b, t, d = x.shape
        # mask shape [B, T], first token always kept
        keep = torch.rand(b, t, device=x.device) > self.p
        keep[:, 0] = 1.0
        keep = keep.unsqueeze(-1)  # [B, T, 1]
        # scale by 1/(1-p) to preserve magnitude
        scale = 1.0 / (1.0 - self.p)
        return x * keep * scale


class TabTransformerGated(nn.Module):
    """
    TabTransformer++ model with gated fusion.

    Parameters
    ----------
    vocab_sizes : Sequence[int]
        A sequence containing the vocabulary size (number of bins) for each
        feature.  Each feature will have its own embedding and MLP.
    config : TTConfig, optional
        Hyperparameters controlling embedding dimensions, number of heads,
        layers, dropout rates, etc.
    """

    def __init__(self, vocab_sizes: Sequence[int], config: TTConfig | None = None):
        super().__init__()
        self.config = config or TTConfig()
        self.num_tokens = len(vocab_sizes)

        # Token embeddings
        self.embs = nn.ModuleList([
            nn.Embedding(v + 1, self.config.emb_dim) for v in vocab_sizes
        ])
        # Per‑feature value MLPs
        self.val_mlps = nn.ModuleList([
            PerTokenValMLP(self.config.emb_dim) for _ in vocab_sizes
        ])
        # Learnable gates (one per feature), initialised to -2.0 for safe start
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor([-2.0])) for _ in vocab_sizes
        ])
        self.sigmoid = nn.Sigmoid()
        # CLS token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.emb_dim))
        # Dropouts
        self.emb_dropout = nn.Dropout(self.config.emb_dropout)
        self.tokendrop = TokenDrop(self.config.tokendrop_p)
        # Transformer encoder with Pre‑LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.emb_dim,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.emb_dim * self.config.ffn_multiplier,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.config.n_layers)
        # Prediction head: LN → Linear → GELU → Dropout → Linear
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.emb_dim),
            nn.Linear(self.config.emb_dim, self.config.emb_dim * 3),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.emb_dim * 3, 1),
        )

    def forward(self, x_tok: torch.Tensor, x_val: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_tok : torch.LongTensor of shape (B, T)
            Quantile‑binned tokens for each feature.
        x_val : torch.FloatTensor of shape (B, T)
            Continuous values for each feature (normalized/scaled).
        Returns
        -------
        torch.Tensor of shape (B,)
            Predicted residuals.
        """
        b = x_tok.size(0)
        # Compute embeddings per feature with gated fusion
        emb_list: List[torch.Tensor] = []
        for i in range(self.num_tokens):
            tok_e = self.embs[i](x_tok[:, i])  # [B, D]
            val_e = self.val_mlps[i](x_val[:, i:i+1])  # [B, D]
            g = self.sigmoid(self.gates[i])  # scalar
            emb_list.append(tok_e + g * val_e)
        # Shape: [B, T, D]
        x = torch.stack(emb_list, dim=1)
        x = self.emb_dropout(x)
        # Prepend CLS token
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)
        # TokenDrop and Transformer
        x = self.tokendrop(x)
        x = self.encoder(x)
        # Extract CLS token and predict
        return self.head(x[:, 0, :]).squeeze(-1)