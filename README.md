# TabTransformer++ for Residual Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>A novel extension of TabTransformer with gated fusion for residual-based model stacking</strong>
</p>

---

## Overview

This project implements **TabTransformer++**, an enhanced transformer architecture designed specifically for tabular data in a **residual learning** framework. Rather than predicting targets directly, the model learns to correct errors from simpler base models—a powerful technique for competition-winning ensembles.

### The Residual Learning Approach

```
+-------------------+     +------------------------+     +-----------------------+
|    Base Model     |     |    TabTransformer++    |     |   Final Prediction    |
| (Ridge, XGBoost)  | --> |   Predicts Residual    | --> |   Base + Residual     |
|    -> base_pred   |     |        (error)         |     |                       |
+-------------------+     +------------------------+     +-----------------------+
```

**Why residual learning?**
- Base models capture linear/tree patterns efficiently
- Transformers excel at learning complex feature interactions
- Combined: each model focuses on what it does best

---

## Novel Architectural Contributions

TabTransformer++ introduces six key innovations over the original TabTransformer:

### 1. Dual Representation (Tokens + Scalars)

Each feature is represented in two complementary ways:

| Type | Creation | Captures |
|------|----------|----------|
| **Token Embedding** | Quantile bin -> learned vector | Discrete patterns, ordinal relationships |
| **Value Embedding** | Raw scalar -> MLP projection | Precise numeric magnitude |

*Why both?* Binning loses precision (1.01 and 1.99 may share a bin), but raw scalars lack pattern-matching power.

### 2. Learnable Gated Fusion

Per-feature gates (initialized to 0) control the blend:

```python
final_emb[i] = token_emb[i] + sigmoid(gate[i]) * value_emb[i]
```

- Gates are **learned independently** for each feature
- Model adapts to each column's characteristics automatically

### 3. Per-Token Value MLPs

Each feature gets its own projection network instead of sharing:

```
Linear(1 -> 64) -> GELU -> Linear(64 -> 64) -> LayerNorm
```

Allows different transformations for different feature distributions.

### 4. TokenDrop Regularization

During training, randomly zero out feature embeddings (p=0.12):

```python
mask = (random > p)   # per-sample, per-feature
mask[:, 0] = 1.0      # Never drop CLS token
x = x * mask
```

Prevents over-reliance on any single feature.

### 5. CLS Token Aggregation

BERT-style `[CLS]` token prepended to the sequence:

```
[CLS, feat_1, feat_2, ..., feat_n, base_pred, dt_pred]
```

CLS attends to all features and produces the final representation.

### 6. Pre-LayerNorm Transformer

Uses `norm_first=True` for more stable training without warmup:

```
Pre-LN:  x = x + Attention(LayerNorm(x))   [Stable]
Post-LN: x = LayerNorm(x + Attention(x))   [Requires warmup]
```

---

## Architecture Diagram

```
                     +-------------------------------------+
                     |     INPUT: T features + 2 meta     |
                     |   (tokens, raw_values) per feature |
                     +-------------------------------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
          v                            v                            v
   +-------------+              +-------------+              +-------------+
   | Feature 1   |              | Feature 2   |     ...      | Feature T   |
   | token->embed|              | token->embed|              | token->embed|
   | value->MLP  |              | value->MLP  |              | value->MLP  |
   | gate fusion |              | gate fusion |              | gate fusion |
   +-------------+              +-------------+              +-------------+
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                                       v
                            +----------------------+
                            |  Embedding Dropout   |
                            |      (p=0.05)        |
                            +----------------------+
                                       |
                                       v
                            +----------------------+
                            |   Prepend [CLS]      |
                            |      Token           |
                            +----------------------+
                                       |
                                       v
                            +----------------------+
                            |    TokenDrop         |
                            |  (p=0.12, train)     |
                            +----------------------+
                                       |
                                       v
                     +-------------------------------------+
                     |      TRANSFORMER ENCODER            |
                     |  +-------------------------------+  |
                     |  | Layer 1: 4-head attention     |  |
                     |  | + FFN(64->256->64) + PreLN    |  |
                     |  +-------------------------------+  |
                     |  +-------------------------------+  |
                     |  | Layer 2: 4-head attention     |  |
                     |  | + FFN(64->256->64) + PreLN    |  |
                     |  +-------------------------------+  |
                     |  +-------------------------------+  |
                     |  | Layer 3: 4-head attention     |  |
                     |  | + FFN(64->256->64) + PreLN    |  |
                     |  +-------------------------------+  |
                     +-------------------------------------+
                                       |
                                       v
                            +----------------------+
                            |  Extract [CLS]       |
                            |    Embedding         |
                            +----------------------+
                                       |
                                       v
                     +-------------------------------------+
                     |         PREDICTION HEAD             |
                     |  LayerNorm -> Linear(64->192)       |
                     |  -> GELU -> Dropout -> Linear(192->1)|
                     +-------------------------------------+
                                       |
                                       v
                          +---------------------+
                          | Predicted Residual  |
                          |    (z-scored)       |
                          +---------------------+
```

---

## Training Pipeline

The notebook implements a complete 5-fold cross-validation pipeline:

### Step 1: Base Model Stacking
```python
# Ridge regression for base predictions
model_base = Ridge(alpha=1.0)

# RandomForest for additional signal  
model_dt = RandomForestRegressor(n_estimators=20, max_depth=8)

# Out-of-fold predictions to prevent leakage
residual = target - base_pred
```

### Step 2: Tabular Tokenization
- **Quantile binning**: 32 bins for features, 128 for base_pred, 64 for tree_pred
- **Z-score normalization**: Preserves raw numeric information
- Fit on training fold only (leak-free)

### Step 3: Train TabTransformer++
- **EMA (Polyak averaging)**: Maintains exponential moving average of weights
- **Huber loss**: Robust to outliers
- **AdamW optimizer**: With weight decay regularization

### Step 4: Isotonic Calibration
Post-training calibration maps z-scored predictions to actual residuals:
```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(preds_z, y_va_raw)
calibrated = iso.predict(preds_z)
```

### Step 5: Final Ensemble
```python
final_prediction = base_pred + calibrated_residual
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/LEDazzio01/Tab-Transformer-Residual-Learning.git
cd Tab-Transformer-Residual-Learning

# Install dependencies
pip install numpy pandas torch scikit-learn jupyter
```

---

## Quick Start

```bash
# Launch the notebook
jupyter notebook TabTransformer_Residual_Learning.ipynb
```

The notebook is self-contained and demonstrates the full pipeline using the **California Housing** dataset.

---

## Hyperparameters

All hyperparameters are centralized in the `Config` class:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Tokenization** | `NBINS` | 32 | Quantile bins for numeric features |
| | `NBINS_BASE` | 128 | Bins for base model predictions |
| | `NBINS_DT` | 64 | Bins for tree model predictions |
| **Architecture** | `EMB_DIM` | 64 | Embedding dimension (d_model) |
| | `N_HEADS` | 4 | Multi-head attention heads |
| | `N_LAYERS` | 3 | Transformer encoder layers |
| | `MLP_HID` | 192 | Prediction head hidden dim |
| **Regularization** | `DROPOUT` | 0.1 | Attention & FFN dropout |
| | `EMB_DROPOUT` | 0.05 | Post-embedding dropout |
| | `TOKENDROP_P` | 0.12 | TokenDrop probability |
| **Training** | `EPOCHS` | 10 | Training epochs |
| | `BATCH_SIZE` | 1024 | Batch size |
| | `LR` | 2e-3 | AdamW learning rate |
| | `EMA_DECAY` | 0.995 | Polyak averaging decay |

---

## File Structure

```
Tab-Transformer-Residual-Learning/
├── README.md                              # This file
├── TabTransformer_Residual_Learning.ipynb # Complete implementation
└── ...
```

---

## Citation

If you use this code, please cite the original TabTransformer paper:

```bibtex
@article{huang2020tabtransformer,
  title={TabTransformer: Tabular Data Modeling Using Contextual Embeddings},
  author={Huang, Xin and Khetan, Ashish and Cvitkovic, Milan and Karnin, Zohar},
  journal={arXiv preprint arXiv:2012.06678},
  year={2020}
}
```

---

## Acknowledgments

- **[TabTransformer Paper](https://arxiv.org/abs/2012.06678)** — Huang et al. (2020)
- **[tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)** — Reference implementation by lucidrains
- **[California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)** — Demo dataset from scikit-learn

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
