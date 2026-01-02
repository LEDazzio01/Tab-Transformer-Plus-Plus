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

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#novel-architectural-contributions">Architecture</a> •
  <a href="#system-design-production-deployment">Production</a> •
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

## Overview

This project implements **TabTransformer++**, an enhanced transformer architecture designed specifically for tabular data in a **residual learning** framework. Rather than predicting targets directly, the model learns to correct errors from simpler base models—a powerful technique for competition-winning ensembles.

### The Residual Learning Approach

```
+--------------------+     +------------------------+     +-----------------------+
|    Base Model      |     |    TabTransformer++    |     |   Final Prediction    |
| (HistGBR, XGBoost) | --> |   Predicts Residual    | --> |   Base + Residual     |
|    -> base_pred    |     |        (error)         |     |                       |
+--------------------+     +------------------------+     +-----------------------+
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

### 2. Learnable Gated Fusion (Safe Initialization)

Per-feature gates control the blend between token and scalar representations:

```python
final_emb[i] = token_emb[i] + sigmoid(gate[i]) * value_emb[i]
```

**Safe Initialization**: Gates are initialized to **-2.0** (sigmoid ≈ 0.12), biasing the model to rely on stable token embeddings first. This prevents early divergence before the model learns when to trust scalar values.

- Gates are **learned independently** for each feature
- Model adapts to each column's characteristics automatically
- Low gate → token-dominant (categorical treatment)
- High gate → scalar-dominant (precise numeric treatment)

### 3. Per-Token Value MLPs

Each feature gets its own projection network instead of sharing:

```
Linear(1 -> 64) -> GELU -> Linear(64 -> 64) -> LayerNorm
```

Allows different transformations for different feature distributions.

### 4. TokenDrop Regularization (with Inverted Scaling)

During training, randomly zero out feature embeddings (p=0.12):

```python
mask = (random > p)   # per-sample, per-feature
mask[:, 0] = 1.0      # Never drop CLS token
x = x * mask / (1 - p)  # Inverted scaling for magnitude consistency
```

Prevents over-reliance on any single feature. The inverted scaling maintains expected magnitude between train and test modes (like standard Dropout).

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
                          |  (robust-scaled)    |
                          +---------------------+
```

---

## Interpretability Features

TabTransformer++ includes built-in interpretability tools:

### Gate Value Visualization

Extract and visualize learned gate values to understand feature treatment:

```python
gate_values = extract_gate_values(model, feature_names)
visualize_gate_values(gate_values)
```

- **Low gate (near 0)**: Feature works better as categorical bins
- **High gate (near 1)**: Feature requires precise scalar values

### Token Embedding Visualization

Visualize learned embeddings using t-SNE or PCA:

```python
visualize_token_embeddings(model, tokenizer, feature_idx=0, method='pca')
```

Shows how the model organizes quantile bins in embedding space, revealing learned semantic relationships.

---

## Why TabTransformer++ Over XGBoost?

Even when RMSE is comparable, TabTransformer++ offers unique advantages:

| Capability | XGBoost | TabTransformer++ |
|------------|---------|------------------|
| **Dense Embeddings** | ❌ No | ✅ Each row becomes a learned vector |
| **Multi-Modal Fusion** | ❌ Cannot combine with images/text | ✅ Embeddings fuse with vision/NLP models |
| **Transfer Learning** | ❌ Must retrain from scratch | ✅ Pre-train on large tables, fine-tune on small |
| **Interpretable Gates** | ❌ Feature importance only | ✅ Learn token vs scalar preference per feature |
| **GPU Batch Inference** | ⚠️ Limited | ✅ Native PyTorch batching |

**The Real Value**: TabTransformer++ generates **dense embeddings** suitable for downstream multi-modal tasks (e.g., combining tabular property data with house images).

---

## Training Pipeline

The notebook implements a complete 5-fold cross-validation pipeline:

### Step 1: Base Model Stacking
```python
# HistGradientBoostingRegressor for base predictions (captures non-linearity)
model_base = HistGradientBoostingRegressor(max_iter=100, max_depth=5)

# RandomForest for additional signal  
model_dt = RandomForestRegressor(n_estimators=20, max_depth=8)

# Out-of-fold predictions to prevent leakage
residual = target - base_pred
```

**Why HistGradientBoostingRegressor instead of Ridge?**
- Captures non-linear patterns that linear models miss
- Leaves purer high-order feature interactions for the Transformer
- Faster than RandomForest due to histogram-based splits

### Step 2: Tabular Tokenization
- **Quantile binning**: 32 bins for features, 128 for base_pred, 64 for tree_pred
- **Robust scaling**: `(x - median) / IQR` — resistant to outliers (replaces Z-score)
- Fit on training fold only (leak-free)

**Why Robust Scaling?** Z-score `(x - mean) / std` is sensitive to outliers, which can cause gradient explosions in the scalar path. Robust scaling using median and IQR stabilizes training across all folds.

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

## System Design: Production Deployment

This section outlines how TabTransformer++ fits into a production ML system.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────────┐  │
│   │  Raw Data    │───▶│  TabularTokenizer │───▶│  Feature Store          │  │
│   │  (Offline)   │    │  .fit() on TRAIN  │    │  (Serialize tokenizer)  │  │
│   └──────────────┘    └───────────────────┘    └─────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│                     ┌─────────────────────┐                                  │
│                     │  TabTransformer++   │                                  │
│                     │  PyTorch Training   │                                  │
│                     └─────────────────────┘                                  │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Model Export                                     │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │  • torch.jit.script() → TorchScript (.pt)                           │   │
│   │  • torch.onnx.export() → ONNX (.onnx)                               │   │
│   │  • TensorRT optimization for NVIDIA GPUs                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────────┐  │
│   │  New Request │───▶│  Feature Store    │───▶│  Tokenizer.transform()  │  │
│   │  (Online)    │    │  (Load tokenizer) │    │  (Consistent binning)   │  │
│   └──────────────┘    └───────────────────┘    └─────────────────────────┘  │
│                                                          │                   │
│                                                          ▼                   │
│                              ┌────────────────────────────────────────────┐  │
│                              │  Inference Runtime                         │  │
│                              ├────────────────────────────────────────────┤  │
│                              │  • ONNX Runtime (CPU/GPU)                  │  │
│                              │  • TensorRT (NVIDIA, <1ms latency)         │  │
│                              │  • TorchServe / Triton Inference Server    │  │
│                              └────────────────────────────────────────────┘  │
│                                                          │                   │
│                                                          ▼                   │
│                              ┌─────────────────────────────────────────┐     │
│                              │  Prediction + Post-Processing           │     │
│                              │  base_pred + calibrated_residual        │     │
│                              └─────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Production Considerations

#### 1. Tokenizer Serialization to Feature Store

The `TabularTokenizer` encapsulates learned quantile bins and scaling statistics. For online/offline consistency:

```python
import pickle

# After training
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Upload to Feature Store (e.g., Feast, Tecton, SageMaker Feature Store)
feature_store.register_artifact("tabtransformer_tokenizer", "tokenizer.pkl")
```

**Why Feature Store?**
- Ensures identical preprocessing in training and serving
- Version control for tokenizer artifacts
- Supports A/B testing different tokenizer configurations

#### 2. Model Export for Low-Latency Inference

```python
# Export to ONNX (cross-platform, optimized inference)
import torch.onnx

model.eval()
dummy_tok = torch.randint(0, 32, (1, num_features))
dummy_val = torch.randn(1, num_features)

torch.onnx.export(
    model,
    (dummy_tok, dummy_val),
    "tabtransformer.onnx",
    input_names=["tokens", "values"],
    output_names=["prediction"],
    dynamic_axes={"tokens": {0: "batch"}, "values": {0: "batch"}},
)

# For NVIDIA GPUs: Convert to TensorRT
# trtexec --onnx=tabtransformer.onnx --saveEngine=tabtransformer.trt --fp16
```

**Inference Latency Targets:**
| Runtime | Hardware | Typical Latency |
|---------|----------|-----------------|
| PyTorch | CPU | 5-20ms |
| ONNX Runtime | CPU | 2-8ms |
| ONNX Runtime | GPU | 0.5-2ms |
| TensorRT | NVIDIA GPU | <1ms |

#### 3. Online vs. Offline Feature Consistency

**Problem**: Training uses batch statistics; serving sees single rows.

**Solution**: Store computed features, don't recompute at inference.

| Feature Type | Training | Serving |
|--------------|----------|---------|
| Raw features | Compute from source | Fetch from Feature Store |
| Base model predictions | OOF predictions | Pre-computed daily batch |
| Tokenized features | Batch transform | Single-row transform |

**Preventing Train-Serve Skew:**
1. **Tokenizer versioning**: Hash tokenizer params, embed in model metadata
2. **Feature validation**: Assert feature distributions at inference time
3. **Shadow mode**: Run new model in parallel, compare outputs before deployment

#### 4. Deployment Architecture Options

**Option A: Batch Prediction (Offline)**
```
Airflow/Prefect → Load Data → Transform → Predict → Write to DB
```
- Use for: Daily scoring of large datasets
- Latency: Hours (acceptable)
- Cost: Low (spot instances)

**Option B: Real-Time API (Online)**
```
API Gateway → Load Balancer → Inference Pod (ONNX/TensorRT) → Response
```
- Use for: User-facing predictions
- Latency: <50ms p99
- Scaling: Horizontal pod autoscaling

**Option C: Streaming (Near Real-Time)**
```
Kafka → Feature Compute → Model Inference → Kafka → Downstream
```
- Use for: Event-driven predictions
- Latency: Seconds
- Throughput: High (parallelizable)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/LEDazzio01/Tab-Transformer-Plus-Plus.git
cd Tab-Transformer-Plus-Plus

# Install dependencies
pip install numpy pandas torch scikit-learn jupyter
```

---

## Quick Start

### Option 1: Command-Line Interface

```bash
# Install the package
pip install -e .

# Train on built-in California Housing dataset
ttpp train --dataset cal_housing --epochs 10 --batch_size 1024

# Train on your own CSV data
ttpp train --train_data data/train.csv --target_col price --epochs 20

# Train with explicit train/test split
ttpp train --train_data train.csv --test_data test.csv --target_col target --n_folds 5
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook TabTransformer_Residual_Learning.ipynb
```

The notebook demonstrates the full pipeline using the **California Housing** dataset.

### Option 3: Python API

```python
import pandas as pd
from tab_transformer_plus_plus.model import TabTransformerGated, TTConfig
from tab_transformer_plus_plus.tokenizer import TabularTokenizer

# Load your data
df = pd.read_csv("data.csv")
train_df, test_df = train_test_split(df, test_size=0.2)

# Fit tokenizer on TRAINING data only (prevents leakage)
tokenizer = TabularTokenizer(n_bins=32, features=feature_cols, target="target")
tokenizer.fit(train_df)  # Never fit on full dataset!

# Transform data
X_train_tok, X_train_val = tokenizer.transform(train_df)
X_test_tok, X_test_val = tokenizer.transform(test_df)

# Create and train model
model = TabTransformerGated(vocab_sizes=tokenizer.get_vocab_sizes())
# ... training loop
```

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
Tab-Transformer-Plus-Plus/
├── README.md                              # This documentation
├── CHANGELOG.md                           # Version history and changes
├── LICENSE                                # MIT License
├── pyproject.toml                         # Package configuration
├── requirements.txt                       # Dependencies
├── TabTransformer_Residual_Learning.ipynb # Interactive notebook demo
├── src/
│   └── tab_transformer_plus_plus/
│       ├── __init__.py                    # Package exports
│       ├── model.py                       # TabTransformerGated model (vectorized)
│       ├── tokenizer.py                   # TabularTokenizer (optimized)
│       ├── train.py                       # Training pipeline & CLI
│       └── utils.py                       # Utility functions
└── tests/
    └── test_model.py                      # Unit & sanity tests
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
