"""Tokenizer for TabTransformer++.

This module implements a dual‑representation tokenizer for tabular data.
Each continuous feature is quantile‑binned to produce a discrete token and
robust‑scaled (median/IQR) to produce a continuous value.  The tokenizer
stores bin edges and scaling statistics, and can transform new data using
the learned parameters.
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

        Parameters
        ----------
        df : pandas.DataFrame
            Training dataframe containing the features and target.

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
        self.bin_edges_ = []
        self.medians_ = []
        self.iqrs_ = []
        self.vocab_sizes_ = []
        for col in self.features:
            col_values = df[col].values.astype(float)
            # Compute quantile edges (n_bins equi‑depth bins)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(col_values, quantiles)
            # Ensure strict monotonic edges (np.quantile may produce duplicates)
            # Add small epsilon to avoid identical edges
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6
            self.bin_edges_.append(edges)
            # Compute median and IQR for robust scaling
            median = np.median(col_values)
            q1 = np.quantile(col_values, 0.25)
            q3 = np.quantile(col_values, 0.75)
            iqr = q3 - q1
            # avoid zero IQR
            if iqr == 0:
                iqr = 1.0
            self.medians_.append(median)
            self.iqrs_.append(iqr)
            # Vocabulary size equals number of bins
            self.vocab_sizes_.append(self.n_bins)
        # Store target stats if provided
        if self.target is not None and self.target in df.columns:
            y = df[self.target].values
            self.target_mu_ = float(np.mean(y))
            self.target_std_ = float(np.std(y)) if np.std(y) > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a dataframe into token and scalar arrays.

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
        x_tok = np.zeros((n_samples, n_features), dtype=np.int64)
        x_val = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, col in enumerate(self.features):
            values = df[col].values.astype(float)
            edges = self.bin_edges_[i]
            # Digitize values into bins (right‑most bin has index n_bins‑1)
            # np.digitize returns indices 1..n_bins for bin membership
            inds = np.digitize(values, edges[1:-1], right=False)
            x_tok[:, i] = inds
            # Robust scaling: (x - median) / IQR
            x_val[:, i] = (values - self.medians_[i]) / self.iqrs_[i]
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