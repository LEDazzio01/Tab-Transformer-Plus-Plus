"""Training pipeline and CLI for TabTransformer++.

This module implements a simple training pipeline for TabTransformer++ on
tabular datasets.  It includes a command‑line interface that can train
the model on the California Housing dataset.

Usage:

```bash
python -m tab_transformer_plus_plus.train train cal_housing --epochs 10 --batch_size 1024
```

The `ttpp` entrypoint exposed via `pyproject.toml` forwards to this
module's `main()` function.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

from .model import TabTransformerGated, TTConfig
from .tokenizer import TabularTokenizer
from .utils import seed_everything, root_rmse


def _train_tab_transformer(
    X_tr_tok: np.ndarray,
    X_tr_val: np.ndarray,
    y_tr: np.ndarray,
    X_va_tok: np.ndarray,
    X_va_val: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 1024,
    device: str = "cpu",
    config: TTConfig | None = None,
) -> Tuple[np.ndarray, TabTransformerGated]:
    """Train TabTransformerGated on a single fold and return predictions for the validation set.

    This helper trains the model for a given number of epochs and returns
    the predicted residuals on the validation data.  A new model is
    instantiated for each fold and not reused across folds.

    Parameters
    ----------
    X_tr_tok, X_tr_val : numpy.ndarray
        Tokenised training and validation inputs (integer tokens).
    y_tr : numpy.ndarray
        Z‑scored residuals for training.
    X_va_tok, X_va_val : numpy.ndarray
        Tokenised validation inputs (integer tokens and scaled continuous values).
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for the PyTorch DataLoader.
    device : str
        Device to use ('cpu' or 'cuda').
    config : TTConfig | None
        Model configuration.  If None, defaults are used.

    Returns
    -------
    preds : numpy.ndarray
        Predictions on the validation set (z‑score space).
    model : TabTransformerGated
        The trained model instance (last epoch).
    """
    torch_device = torch.device(device)
    dataset_tr = list(zip(X_tr_tok, X_tr_val, y_tr))
    dataset_va = list(zip(X_va_tok, X_va_val))
    model = TabTransformerGated(vocab_sizes=np.max(X_tr_tok, axis=0) + 1, config=config)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
    model.train()
    for epoch in range(n_epochs):
        # Shuffle training dataset each epoch
        np.random.shuffle(dataset_tr)
        for i in range(0, len(dataset_tr), batch_size):
            batch = dataset_tr[i : i + batch_size]
            x_tok_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=torch_device)
            x_val_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=torch_device)
            y_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=torch_device)
            optimizer.zero_grad()
            pred = model(x_tok_batch, x_val_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    # Prediction on validation set
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(dataset_va), batch_size):
            batch = dataset_va[i : i + batch_size]
            x_tok_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=torch_device)
            x_val_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=torch_device)
            preds = model(x_tok_batch, x_val_batch)
            preds_list.append(preds.cpu().numpy())
    return np.concatenate(preds_list), model


def _fit_base_models(X: np.ndarray, y: np.ndarray, cv: KFold) -> Tuple[np.ndarray, np.ndarray]:
    """Fit base models with cross‑validation and return out‑of‑fold and test predictions.

    The base models are histogram‑based gradient boosting and random forest.
    """
    oof_base = np.zeros(len(X))
    oof_dt = np.zeros(len(X))
    base_preds_test = []
    dt_preds_test = []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        # HistGradientBoostingRegressor for base predictions
        model_base = HistGradientBoostingRegressor(max_iter=100, max_depth=5)
        model_base.fit(X_tr, y_tr)
        oof_base[va_idx] = model_base.predict(X_va)
        # RandomForestRegressor for tree predictions
        model_dt = RandomForestRegressor(n_estimators=20, max_depth=8)
        model_dt.fit(X_tr, y_tr)
        oof_dt[va_idx] = model_dt.predict(X_va)
        base_preds_test.append(model_base)
        dt_preds_test.append(model_dt)
    # Fit models on full training set to prepare test predictors
    base_agg = HistGradientBoostingRegressor(max_iter=100, max_depth=5)
    dt_agg = RandomForestRegressor(n_estimators=20, max_depth=8)
    base_agg.fit(X, y)
    dt_agg.fit(X, y)
    return oof_base, oof_dt


def train_cal_housing(epochs: int = 10, batch_size: int = 1024, seed: int = 2025) -> None:
    """Train TabTransformer++ on the California Housing dataset.

    This function performs a 5‑fold cross‑validation training loop that
    combines base models with TabTransformer++.  It prints RMSE for the
    base model alone and for the residual model.
    """
    seed_everything(seed)
    # Load data
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    # Split into train/test to evaluate holdout performance
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    target_col = "MedHouseVal"
    features = [c for c in df.columns if c != target_col]
    # Prepare cross-validation for base models
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    # Fit base models and get OOF predictions
    oof_base, oof_dt = _fit_base_models(X_train, y_train, cv)
    # Fit base models on full training to get test predictions
    model_base_full = HistGradientBoostingRegressor(max_iter=100, max_depth=5)
    model_dt_full = RandomForestRegressor(n_estimators=20, max_depth=8)
    model_base_full.fit(X_train, y_train)
    model_dt_full.fit(X_train, y_train)
    test_base = model_base_full.predict(test_df[features])
    test_dt = model_dt_full.predict(test_df[features])
    # Add predictions to dataframes
    train_df = train_df.assign(base_pred=oof_base, tree_pred=oof_dt)
    test_df = test_df.assign(base_pred=test_base, tree_pred=test_dt)
    # Compute residuals for training
    train_df = train_df.assign(residual=train_df[target_col] - train_df.base_pred)
    # Cross‑validated training of TabTransformer
    oof_residual = np.zeros(len(train_df))
    cv_rmse = []
    folds = cv.split(train_df)
    fold_idx = 0
    for tr_indices, va_indices in cv.split(train_df):
        fold_idx += 1
        tr_df = train_df.iloc[tr_indices].reset_index(drop=True)
        va_df = train_df.iloc[va_indices].reset_index(drop=True)
        # Define tokenizer with features + predictions
        tokenizer = TabularTokenizer(
            n_bins=32,
            features=features + ["base_pred", "tree_pred"],
            target="residual",
        )
        tokenizer.fit(tr_df)
        # Transform train and val
        X_tr_tok, X_tr_val = tokenizer.transform(tr_df)
        X_va_tok, X_va_val = tokenizer.transform(va_df)
        # Z‑score residuals
        y_tr = (tr_df["residual"].values - tokenizer.target_mu_) / tokenizer.target_std_
        y_va_raw = va_df["residual"].values
        # Train transformer
        preds_z, model = _train_tab_transformer(
            X_tr_tok,
            X_tr_val,
            y_tr,
            X_va_tok,
            X_va_val,
            n_epochs=epochs,
            batch_size=batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Convert back from z‑scores
        preds_raw = preds_z * tokenizer.target_std_ + tokenizer.target_mu_
        oof_residual[va_indices] = preds_raw
        rmse = root_rmse(y_va_raw, preds_raw)
        cv_rmse.append(rmse)
        print(f"Fold {fold_idx} | Residual RMSE: {rmse:.4f}")
    # Train transformer on full training set to predict test residuals
    tokenizer_full = TabularTokenizer(
        n_bins=32,
        features=features + ["base_pred", "tree_pred"],
        target="residual",
    )
    tokenizer_full.fit(train_df)
    X_train_tok, X_train_val = tokenizer_full.transform(train_df)
    y_train_z = (train_df["residual"].values - tokenizer_full.target_mu_) / tokenizer_full.target_std_
    X_test_tok, X_test_val = tokenizer_full.transform(test_df)
    preds_z_full, model_full = _train_tab_transformer(
        X_train_tok,
        X_train_val,
        y_train_z,
        X_test_tok,
        X_test_val,
        n_epochs=epochs,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    preds_residual_test = preds_z_full * tokenizer_full.target_std_ + tokenizer_full.target_mu_
    # Combine base and residual predictions
    train_pred = train_df.base_pred.values + oof_residual
    test_pred = test_df.base_pred.values + preds_residual_test
    # Compute RMSE for training and test
    base_rmse_train = root_rmse(train_df[target_col].values, train_df.base_pred.values)
    model_rmse_train = root_rmse(train_df[target_col].values, train_pred)
    base_rmse_test = root_rmse(test_df[target_col].values, test_df.base_pred.values)
    model_rmse_test = root_rmse(test_df[target_col].values, test_pred)
    print("\nFinal Results (RMSE):")
    print(f"  Base Model Only (train):           {base_rmse_train:.5f}")
    print(f"  Base + TabTransformer++ (train):   {model_rmse_train:.5f}")
    print(f"  Base Model Only (test):            {base_rmse_test:.5f}")
    print(f"  Base + TabTransformer++ (test):    {model_rmse_test:.5f}")


def main(argv: List[str] | None = None) -> None:
    """Entry point for the TabTransformer++ CLI.

    Currently supports only the `train` command on the California Housing
    dataset.  Additional subcommands and datasets can be added in the
    future.
    """
    parser = argparse.ArgumentParser(description="TabTransformer++ CLI")
    subparsers = parser.add_subparsers(dest="command")
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train TabTransformer++ on a dataset")
    train_parser.add_argument(
        "dataset",
        choices=["cal_housing"],
        help="Dataset to train on (currently only California Housing)",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    train_parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.dataset == "cal_housing":
            train_cal_housing(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])