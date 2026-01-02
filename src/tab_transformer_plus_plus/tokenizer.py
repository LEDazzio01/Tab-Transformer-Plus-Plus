"""Tokenizer for TabTransformer++.

This module implements a dual‑representation tokenizer for tabular data.
Each continuous feature is quantile‑binned to produce a discrete token and
robust‑scaled (median/IQR) to produce a continuous value.  The tokenizer
stores bin edges and scaling statistics, and can transform new data using
the learned parameters.

IMPORTANT: Data Leakage Prevention
----------------------------------
The TabularTokenizer computes global quantile bins and scaling statistics
(medians, IQRs) from the data passed to fit(). To prevent target leakage:

1. **ONLY fit on training data** - Never fit on the full dataset before
   train/test split. The test set must be unseen during tokenizer fitting.

2. **Use separate tokenizers per fold** in cross-validation - Each fold
   should have its own tokenizer fitted only on that fold's training data.

3. **Serialize the training tokenizer** for inference - The production
   tokenizer must use the same bin edges learned from training data.

Example of CORRECT usage:
    >>> train_df, test_df = train_test_split(df, test_size=0.2)
    >>> tokenizer = TabularTokenizer(n_bins=32, features=feature_cols)
    >>> tokenizer.fit(train_df)  # Fit ONLY on training data
    >>> X_train_tok, X_train_val = tokenizer.transform(train_df)
    >>> X_test_tok, X_test_val = tokenizer.transform(test_df)

Example of INCORRECT usage (causes leakage):
    >>> tokenizer.fit(df)  # WRONG: includes test data in quantile computation
    >>> train_df, test_df = train_test_split(df, test_size=0.2)

Why this matters: Quantile binning uses the full data distribution. If test
data is included, the model indirectly "sees" test distribution during
training, leading to overly optimistic validation metrics that don't
generalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TabularTokenizer:
    """
    Dual‑representation tokenizer for TabTransformer++.

    This tokenizer converts continuous features into two representations:
    1. **Discrete tokens**: Quantile-binned indices for embedding lookup
    2. **Continuous values**: Robust-scaled (median/IQR) for scalar MLPs

    WARNING: Leakage Prevention
    ---------------------------
    This tokenizer MUST ONLY be fit on training data. The fit() method
    computes global quantile bins and scaling statistics that would leak
    information if computed on the full dataset (including validation/test).

    For cross-validation, create a new tokenizer for each fold and fit it
    only on that fold's training portion.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for continuous features.  All features share
        the same number of bins for simplicity.
    features : Iterable[str] | None, optional
        List of feature column names to tokenize.  If None, all columns
        except the target and any metadata are used.
    target : str | None, optional
        Name of the target column.  Used to compute target statistics for
        residual training.

    Attributes
    ----------
    bin_edges_ : List[np.ndarray]
        Learned quantile bin edges for each feature (set after fit).
    medians_ : List[float]
        Learned medians for robust scaling (set after fit).
    iqrs_ : List[float]
        Learned IQRs for robust scaling (set after fit).
    vocab_sizes_ : List[int]
        Vocabulary size (n_bins) for each feature.
    target_mu_ : float | None
        Mean of target column (for z-scoring residuals).
    target_std_ : float | None
        Std of target column (for z-scoring residuals).

    Examples
    --------
    >>> # CORRECT: Fit only on training data
    >>> tokenizer = TabularTokenizer(n_bins=32, target="price")
    >>> tokenizer.fit(train_df)
    >>> X_train_tok, X_train_val = tokenizer.transform(train_df)
    >>> X_test_tok, X_test_val = tokenizer.transform(test_df)

    See Also
    --------
    sklearn.preprocessing.KBinsDiscretizer : Similar binning approach
    """

    n_bins: int = 32
    features: Iterable[str] | None = None
    target: str | None = None
    bin_edges_: List[np.ndarray] = field(default_factory=list, init=False)
    medians_: List[float] = field(default_factory=list, init=False)
    iqrs_: List[float] = field(default_factory=list, init=False)
    vocab_sizes_: List[int] = field(default_factory=list, init=False)
    target_mu_: float | None = field(default=None, init=False)
    target_std_: float | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> "TabularTokenizer":
        """Learn quantile bins and robust scaling statistics from data.

        This method uses vectorized NumPy operations for efficient computation
        on large datasets. The quantile binning is performed using np.quantile
        with broadcasting, which is significantly faster than iterating through
        features one by one for large datasets.

        Parameters
        ----------
        df : pandas.DataFrame
            Training dataframe containing the features and target.
            WARNING: This must be training data only to prevent leakage.

        Returns
        -------
        self : TabularTokenizer
            Fitted tokenizer.
        """
        # Determine feature columns
        if self.features is None:
            candidates = df.columns.tolist()
            if self.target is not None and self.target in candidates:
                candidates.remove(self.target)
            self.features = candidates
        self.features = list(self.features)  # ensure list

        n_features = len(self.features)

        # Extract all feature values as a single 2D array for vectorized processing
        # Shape: (n_samples, n_features)
        all_values = df[self.features].values.astype(np.float64)

        # Compute quantile edges for all features at once using vectorized np.quantile
        # quantiles: shape (n_bins+1,) -> edges: shape (n_bins+1, n_features)
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        all_edges = np.quantile(all_values, quantiles, axis=0)  # (n_bins+1, n_features)

        # Ensure strict monotonic edges for each feature (vectorized)
        # Add small epsilon to avoid identical edges
        for i in range(1, all_edges.shape[0]):
            mask = all_edges[i, :] <= all_edges[i - 1, :]
            all_edges[i, mask] = all_edges[i - 1, mask] + 1e-6

        # Store edges per feature (transpose and convert to list)
        self.bin_edges_ = [all_edges[:, i] for i in range(n_features)]

        # Compute robust scaling stats (median, IQR) in vectorized manner
        # np.median and np.quantile support axis parameter for efficiency
        medians = np.median(all_values, axis=0)  # (n_features,)
        q1 = np.quantile(all_values, 0.25, axis=0)
        q3 = np.quantile(all_values, 0.75, axis=0)
        iqrs = q3 - q1
        # Avoid zero IQR
        iqrs[iqrs == 0] = 1.0

        self.medians_ = medians.tolist()
        self.iqrs_ = iqrs.tolist()
        self.vocab_sizes_ = [self.n_bins] * n_features

        # Store as arrays for fast vectorized transform
        self._medians_arr = medians
        self._iqrs_arr = iqrs
        self._edges_arr = all_edges  # (n_bins+1, n_features)

        # Store target stats if provided
        if self.target is not None and self.target in df.columns:
            y = df[self.target].values
            self.target_mu_ = float(np.mean(y))
            self.target_std_ = float(np.std(y)) if np.std(y) > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a dataframe into token and scalar arrays.

        This method uses fully vectorized NumPy operations for efficient
        transformation of large datasets. The digitization is performed
        using searchsorted which is optimized for batch processing.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the features (and optionally target).

        Returns
        -------
        x_tok : numpy.ndarray of shape (n_samples, n_features)
            Quantile bin indices (0‑based) for each feature.  Values range
            from 0 to n_bins‑1.
        x_val : numpy.ndarray of shape (n_samples, n_features)
            Robust‑scaled continuous values using median/IQR.
        """
        assert self.bin_edges_, "Tokenizer has not been fitted. Call fit() first."

        n_samples = len(df)
        n_features = len(self.features)

        # Extract all feature values at once
        all_values = df[self.features].values.astype(np.float64)

        # Vectorized robust scaling: (X - medians) / iqrs
        # Broadcasting: (n_samples, n_features) - (n_features,) / (n_features,)
        x_val = (all_values - self._medians_arr) / self._iqrs_arr

        # Vectorized digitization using searchsorted
        # For each feature, use the inner bin edges (excluding first and last)
        x_tok = np.zeros((n_samples, n_features), dtype=np.int64)

        # Use vectorized approach: for each feature, searchsorted is already vectorized
        # over the samples dimension. We loop over features but the inner operation is fast.
        # For extremely large datasets with many features, could parallelize with numba/joblib.
        edges_inner = self._edges_arr[1:-1, :]  # (n_bins-1, n_features)

        for i in range(n_features):
            # np.searchsorted is already vectorized over samples
            # edges[1:-1] gives inner edges for digitize-like behavior
            x_tok[:, i] = np.searchsorted(edges_inner[:, i], all_values[:, i], side='right')

        return x_tok, x_val.astype(np.float32)

    def transform_batch(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a numpy array directly (faster for inference).

        This is an optimized version of transform() that skips DataFrame
        overhead and directly processes numpy arrays. Use this for high-
        throughput inference pipelines.

        Parameters
        ----------
        values : numpy.ndarray of shape (n_samples, n_features)
            Raw feature values in the same order as self.features.

        Returns
        -------
        x_tok : numpy.ndarray of shape (n_samples, n_features)
            Quantile bin indices (0‑based).
        x_val : numpy.ndarray of shape (n_samples, n_features)
            Robust‑scaled continuous values.
        """
        assert self.bin_edges_, "Tokenizer has not been fitted. Call fit() first."
        assert values.shape[1] == len(self.features), (
            f"Expected {len(self.features)} features, got {values.shape[1]}"
        )

        values = values.astype(np.float64)
        n_samples, n_features = values.shape

        # Vectorized robust scaling
        x_val = (values - self._medians_arr) / self._iqrs_arr

        # Vectorized digitization
        x_tok = np.zeros((n_samples, n_features), dtype=np.int64)
        edges_inner = self._edges_arr[1:-1, :]

        for i in range(n_features):
            x_tok[:, i] = np.searchsorted(edges_inner[:, i], values[:, i], side='right')

        return x_tok, x_val.astype(np.float32)

    def get_vocab_sizes(self) -> List[int]:
        """Return the vocabulary size (number of bins) for each feature."""
        return list(self.vocab_sizes_)

    @property
    def stats(self) -> dict:
        """Return a dictionary of stored statistics (medians, IQRs, etc.)."""
        return {
            "medians": self.medians_,
            "iqrs": self.iqrs_,
            "target_mu": self.target_mu_,
            "target_std": self.target_std_,
        }