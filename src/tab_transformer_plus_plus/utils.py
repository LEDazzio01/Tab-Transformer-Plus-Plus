"""Utility functions for TabTransformer++."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def root_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (RMSE).

    Parameters
    ----------
    y_true : numpy.ndarray
        Array of ground truth targets.
    y_pred : numpy.ndarray
        Array of predicted values.

    Returns
    -------
    float
        RMSE between predictions and true targets.
    """
    diff = y_true.astype(float) - y_pred.astype(float)
    return float(np.sqrt(np.mean(diff ** 2)))


def extract_gate_values(model: torch.nn.Module, feature_names: Iterable[str]) -> Dict[str, float]:
    """Extract the learned gate values from a trained TabTransformer model.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of `TabTransformerGated` after training.
    feature_names : iterable of str
        Names of the features corresponding to each gate.

    Returns
    -------
    dict
        Mapping from feature name to sigmoid(gate) value (between 0 and 1).
    """
    gate_values = {}
    for name, param, feature in zip(model.gates, model.gates, feature_names):
        # param is a Parameter of shape [1]; take its value and apply sigmoid
        value = float(torch.sigmoid(param).item())
        gate_values[feature] = value
    return gate_values


def visualize_gate_values(gate_values: Dict[str, float], title: str = "Gate Values", figsize=(8, 4)) -> None:
    """Plot gate values as a bar chart.

    This function uses matplotlib to create a horizontal bar chart showing
    how much each feature relies on the scalar path (higher means more
    reliance on continuous values).

    Parameters
    ----------
    gate_values : dict
        Mapping from feature names to gate values (0..1).
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Size of the figure.
    """
    import matplotlib.pyplot as plt

    features = list(gate_values.keys())
    values = [gate_values[f] for f in features]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(features, values)
    ax.set_xlabel("Gate value (sigmoid)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def visualize_token_embeddings(model: torch.nn.Module, feature_idx: int, method: str = "pca", n_components: int = 2, **kwargs) -> None:
    """Visualise token embeddings using PCA or t‑SNE.

    Parameters
    ----------
    model : torch.nn.Module
        Trained `TabTransformerGated` model.
    feature_idx : int
        Index of the feature whose embeddings to visualise.
    method : {'pca', 'tsne'}, optional
        Dimensionality reduction method.
    n_components : int, optional
        Number of dimensions to reduce to.  For t‑SNE 2 or 3 is typical.
    kwargs : dict
        Additional keyword arguments passed to the dimensionality reducer.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Get embedding matrix for the feature (skip padding index at 0)
    emb = model.embs[feature_idx].weight.detach().cpu().numpy()
    # Remove the last bin (padding) if present; optional depending on use
    data = emb
    # Reduce dimensionality
    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    reduced = reducer.fit_transform(data)
    if reduced.shape[1] != 2:
        raise ValueError("Only 2D plots are supported")
    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=40, alpha=0.7)
    plt.title(f"Token Embeddings (feature {feature_idx})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()