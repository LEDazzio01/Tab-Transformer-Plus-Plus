"""Unit tests for TabTransformer++ modules."""

import numpy as np
import pytest
import torch

from tab_transformer_plus_plus.model import TabTransformerGated, TTConfig
from tab_transformer_plus_plus.tokenizer import TabularTokenizer


def test_gate_initialisation():
    """Gates should be initialised to a value whose sigmoid is ~0.12."""
    vocab_sizes = [10, 15]
    model = TabTransformerGated(vocab_sizes=vocab_sizes)
    # Gates are now a single tensor of shape [T] instead of ParameterList
    gate_vals = torch.sigmoid(model.gates).tolist()
    # All gate values should be close to sigmoid(-2.0) ≈ 0.1192
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


# =============================================================================
# Sanity Tests: Shape and Overfitting
# =============================================================================

class TestShapeConsistency:
    """Test that the model handles various batch sizes and feature counts."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 32, 128])
    @pytest.mark.parametrize("num_features", [1, 3, 8, 20])
    def test_forward_shape_consistency(self, batch_size: int, num_features: int):
        """Model output shape should be (B,) for any valid input shape.

        This test runs random noise through the model with varying batch sizes
        and feature counts to ensure no shape mismatches occur.
        """
        # Create vocab sizes (random bins per feature, between 8 and 64)
        rng = np.random.default_rng(42)
        vocab_sizes = rng.integers(8, 64, size=num_features).tolist()

        # Instantiate model with small config for speed
        config = TTConfig(emb_dim=16, n_heads=2, n_layers=1, ffn_multiplier=2)
        model = TabTransformerGated(vocab_sizes=vocab_sizes, config=config)
        model.eval()

        # Create random input tensors
        x_tok = torch.randint(0, min(vocab_sizes), (batch_size, num_features))
        x_val = torch.randn(batch_size, num_features)

        # Forward pass
        with torch.no_grad():
            output = model(x_tok, x_val)

        # Check output shape
        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"

    def test_single_feature_model(self):
        """Model should work with just one feature."""
        model = TabTransformerGated(vocab_sizes=[32])
        model.eval()

        x_tok = torch.randint(0, 32, (4, 1))
        x_val = torch.randn(4, 1)

        with torch.no_grad():
            output = model(x_tok, x_val)

        assert output.shape == (4,)

    def test_large_batch_size(self):
        """Model should handle large batch sizes without memory issues."""
        vocab_sizes = [32, 32, 32, 32, 32]
        config = TTConfig(emb_dim=32, n_heads=2, n_layers=2)
        model = TabTransformerGated(vocab_sizes=vocab_sizes, config=config)
        model.eval()

        batch_size = 1024
        x_tok = torch.randint(0, 32, (batch_size, 5))
        x_val = torch.randn(batch_size, 5)

        with torch.no_grad():
            output = model(x_tok, x_val)

        assert output.shape == (batch_size,)

    def test_different_vocab_sizes_per_feature(self):
        """Each feature can have different vocabulary sizes."""
        vocab_sizes = [8, 16, 32, 64, 128]
        model = TabTransformerGated(vocab_sizes=vocab_sizes)
        model.eval()

        # Tokens must be within valid range for each feature
        x_tok = torch.stack([
            torch.randint(0, v, (10,)) for v in vocab_sizes
        ], dim=1)
        x_val = torch.randn(10, 5)

        with torch.no_grad():
            output = model(x_tok, x_val)

        assert output.shape == (10,)


class TestOverfitting:
    """Test that the model can overfit a tiny dataset (proves architecture can learn)."""

    def test_overfit_tiny_dataset(self):
        """Model should be able to overfit to near-zero loss on 10 samples.

        This is a sanity check that proves the architecture is capable of
        learning. If the model cannot overfit a tiny dataset, there's likely
        a bug in the forward pass or gradient computation.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        # Create a tiny dataset (10 samples, 4 features)
        n_samples = 10
        n_features = 4
        n_bins = 8

        x_tok = torch.randint(0, n_bins, (n_samples, n_features))
        x_val = torch.randn(n_samples, n_features)
        # Create deterministic targets based on sum of values
        y = (x_val.sum(dim=1) * 0.5 + torch.randn(n_samples) * 0.1)

        # Create model with small capacity
        vocab_sizes = [n_bins] * n_features
        config = TTConfig(
            emb_dim=32,
            n_heads=2,
            n_layers=2,
            ffn_multiplier=2,
            dropout=0.0,  # No dropout for overfitting test
            emb_dropout=0.0,
            tokendrop_p=0.0,
        )
        model = TabTransformerGated(vocab_sizes=vocab_sizes, config=config)

        # Train for many epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        model.train()
        initial_loss = None
        final_loss = None

        for epoch in range(200):
            optimizer.zero_grad()
            pred = model(x_tok, x_val)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 199:
                final_loss = loss.item()

        # Assert that loss decreased significantly
        assert final_loss < initial_loss * 0.1, (
            f"Model failed to overfit. Initial loss: {initial_loss:.4f}, "
            f"Final loss: {final_loss:.4f}. Loss should decrease by at least 90%."
        )

        # Assert near-zero loss (model can fit the tiny dataset)
        assert final_loss < 0.1, (
            f"Model could not achieve near-zero loss on tiny dataset. "
            f"Final loss: {final_loss:.4f}. This indicates a potential bug."
        )

    def test_gradient_flow(self):
        """Verify that gradients flow through all model parameters."""
        vocab_sizes = [16, 16]
        config = TTConfig(emb_dim=16, n_heads=2, n_layers=1)
        model = TabTransformerGated(vocab_sizes=vocab_sizes, config=config)

        x_tok = torch.randint(0, 16, (4, 2))
        x_val = torch.randn(4, 2)
        y = torch.randn(4)

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        pred = model(x_tok, x_val)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        # Check that key parameters have non-zero gradients
        assert model.gates.grad is not None, "Gates should have gradients"
        assert model.gates.grad.abs().sum() > 0, "Gates gradients should be non-zero"

        assert model.embedding.weight.grad is not None, "Embedding should have gradients"

        # Check vectorized MLP gradients
        assert model.val_mlp.weight1.grad is not None, "Val MLP weight1 should have gradients"
        assert model.val_mlp.weight2.grad is not None, "Val MLP weight2 should have gradients"


class TestDeviceCompatibility:
    """Test model works on different devices."""

    def test_cpu_inference(self):
        """Model should work on CPU."""
        model = TabTransformerGated(vocab_sizes=[16, 16])
        model.eval()
        model.cpu()

        x_tok = torch.randint(0, 16, (2, 2))
        x_val = torch.randn(2, 2)

        with torch.no_grad():
            output = model(x_tok, x_val)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self):
        """Model should work on CUDA if available."""
        model = TabTransformerGated(vocab_sizes=[16, 16])
        model.eval()
        model.cuda()

        x_tok = torch.randint(0, 16, (2, 2)).cuda()
        x_val = torch.randn(2, 2).cuda()

        with torch.no_grad():
            output = model(x_tok, x_val)

        assert output.device.type == "cuda"