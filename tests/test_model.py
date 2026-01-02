"""Unit tests for TabTransformer++ modules."""

import numpy as np
import torch

from tab_transformer_plus_plus.model import TabTransformerGated, TTConfig
from tab_transformer_plus_plus.tokenizer import TabularTokenizer


def test_gate_initialisation():
    """Gates should be initialised to a value whose sigmoid is ~0.12."""
    vocab_sizes = [10, 15]
    model = TabTransformerGated(vocab_sizes=vocab_sizes)
    gate_vals = [float(torch.sigmoid(g).item()) for g in model.gates]
    # All gate values should be close to sigmoid(-2.0)
    for g in gate_vals:
        assert abs(g - 0.1192) < 0.05


def test_tokenizer_roundtrip():
    """Tokenizer should produce tokens within range and scale values."""
    df = None
    import pandas as pd
    # Create synthetic dataframe
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feat1": rng.normal(loc=0.0, scale=1.0, size=100),
        "feat2": rng.uniform(low=-5, high=5, size=100),
        "target": rng.normal(loc=10.0, scale=2.0, size=100),
    })
    tokenizer = TabularTokenizer(n_bins=8, features=["feat1", "feat2"], target="target")
    tokenizer.fit(df)
    x_tok, x_val = tokenizer.transform(df)
    # Tokens must be integers within [0, n_bins-1]
    assert x_tok.min() >= 0 and x_tok.max() < 8
    # Scaled values should have approximately zero median
    assert abs(np.median(x_val[:, 0])) < 0.1
    assert abs(np.median(x_val[:, 1])) < 0.1