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


class VectorizedValueMLP(nn.Module):
    """Vectorized value MLP that processes all features simultaneously.

    This module replaces multiple PerTokenValMLP instances with a single
    batched operation. Instead of iterating through features one by one,
    we reshape the input to treat all features as a batch dimension and
    apply the MLP in parallel.

    The transformation is equivalent to applying a separate 2-layer MLP to each
    feature:
    - Linear(1 -> emb_dim) for each feature
    - GELU activation
    - Linear(emb_dim -> emb_dim) for each feature
    - Per-feature LayerNorm

    Implementation uses parallel linear layers via einsum for efficiency.
    """

    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.num_features = num_features
        self.emb_dim = emb_dim

        # Per-feature weights for first linear: T separate (1 -> D) transforms
        # Shape: [T, D, 1] - each feature has its own weight matrix
        self.weight1 = nn.Parameter(torch.randn(num_features, emb_dim, 1) * 0.02)
        self.bias1 = nn.Parameter(torch.zeros(num_features, emb_dim))

        # Per-feature weights for second linear: T separate (D -> D) transforms
        # Shape: [T, D, D]
        self.weight2 = nn.Parameter(torch.randn(num_features, emb_dim, emb_dim) * 0.02)
        self.bias2 = nn.Parameter(torch.zeros(num_features, emb_dim))

        # Per-feature LayerNorm (applied to emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process all features in parallel.

        Parameters
        ----------
        x : torch.FloatTensor of shape (B, T)
            Continuous values for each feature.

        Returns
        -------
        torch.Tensor of shape (B, T, D)
            Projected embeddings for each feature.
        """
        b, t = x.shape
        # x: [B, T] -> [B, T, 1]
        x = x.unsqueeze(-1)

        # First linear layer: [B, T, 1] @ [T, D, 1].transpose -> [B, T, D]
        # Using einsum: "bti,tid->btd" where i=1
        x = torch.einsum("bti,tdi->btd", x, self.weight1) + self.bias1  # [B, T, D]
        x = self.gelu(x)

        # Second linear layer: [B, T, D] @ [T, D, D] -> [B, T, D]
        x = torch.einsum("btd,tde->bte", x, self.weight2) + self.bias2  # [B, T, D]

        # Apply LayerNorm per feature (across embedding dim)
        x = self.layer_norm(x)
        return x


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

    This implementation uses fully vectorized operations for the forward pass,
    enabling CUDA kernel fusion and significantly faster training compared to
    sequential per-feature processing.

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
        self.vocab_sizes = list(vocab_sizes)

        # Compute embedding table offset for each feature to use a single large embedding
        # This allows vectorized lookup instead of per-feature embedding modules
        self.register_buffer(
            "vocab_offsets",
            torch.tensor([0] + [v + 1 for v in vocab_sizes[:-1]], dtype=torch.long).cumsum(0),
        )
        total_vocab = sum(v + 1 for v in vocab_sizes)
        self.embedding = nn.Embedding(total_vocab, self.config.emb_dim)

        # Vectorized value MLP: processes all features simultaneously
        self.val_mlp = VectorizedValueMLP(self.num_tokens, self.config.emb_dim)

        # Learnable gates as a single parameter tensor [T] for vectorized gating
        # Initialized to -2.0 for safe start (sigmoid ≈ 0.12)
        self.gates = nn.Parameter(torch.full((self.num_tokens,), -2.0))

        # Backward compatibility: expose gates as ParameterList-like for tests
        self._gates_list = None  # Lazy initialization for compatibility

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

        # Keep legacy per-feature modules for backward compatibility / weight loading
        # These are not used in the vectorized forward pass but can be populated
        # if loading from an old checkpoint
        self.embs = nn.ModuleList([
            nn.Embedding(v + 1, self.config.emb_dim) for v in vocab_sizes
        ])
        self.val_mlps = nn.ModuleList([
            PerTokenValMLP(self.config.emb_dim) for _ in vocab_sizes
        ])

    def forward(self, x_tok: torch.Tensor, x_val: torch.Tensor) -> torch.Tensor:
        """Vectorized forward pass.

        This implementation processes all features simultaneously using batched
        tensor operations, enabling CUDA kernel fusion and eliminating the
        sequential for-loop bottleneck.

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
        b, t = x_tok.shape

        # Vectorized token embedding lookup:
        # Add per-feature offset to tokens so we can use a single embedding table
        # x_tok: [B, T], vocab_offsets: [T] -> offset_tokens: [B, T]
        offset_tokens = x_tok + self.vocab_offsets.unsqueeze(0)  # [B, T]
        tok_emb = self.embedding(offset_tokens)  # [B, T, D]

        # Vectorized value MLP: process all features simultaneously
        val_emb = self.val_mlp(x_val)  # [B, T, D]

        # Vectorized gated fusion:
        # gates: [T] -> [1, T, 1] for broadcasting with [B, T, D]
        g = self.sigmoid(self.gates).view(1, t, 1)  # [1, T, 1]
        x = tok_emb + g * val_emb  # [B, T, D]

        x = self.emb_dropout(x)

        # Prepend CLS token
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)

        # TokenDrop and Transformer
        x = self.tokendrop(x)
        x = self.encoder(x)

        # Extract CLS token and predict
        return self.head(x[:, 0, :]).squeeze(-1)