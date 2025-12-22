# TabTransformer++ for Residual Learning

This repository implements **TabTransformer++**, an enhanced deep learning architecture for tabular data.

Instead of predicting the target variable directly, this model demonstrates a **Residual Learning (Model Stacking)** strategy: a simple linear model produces a baseline prediction, and a Transformer is trained to predict the **error (residual)** of that baseline. The final prediction combines both components and significantly outperforms either model alone.

Based on: **TabTransformer: Tabular Data Modeling Using Contextual Embeddings** (Huang et al., 2020)

---

## 🧠 Core Concept: Residual Stacking

The regression task is modeled as an **additive decomposition**:

\[
\text{Final Prediction} = \text{Base Prediction} + \text{Predicted Residual}
\]

### Workflow
- **Base Model (Ridge):** A fast, interpretable linear model generates the initial prediction.
- **Residual Target:**

\[
\text{Residual} = \text{True Target} - \text{Base Prediction}
\]

- **Correction Model:** TabTransformer++ learns to predict this residual.
- **Inference:** The predicted residual is added back to the base prediction.

This mirrors how human analysts work: start with a simple baseline, then apply structured corrections.

---

## 🚀 Architectural Innovations

Standard Transformers treat tabular features as discrete tokens. **TabTransformer++** introduces several enhancements to better model **continuous numerical data**.

### Innovation Summary



Innovation

Description

1

Dual Representation

Each feature is represented as both a discrete token (quantile binning) and a continuous scalar (z-score normalization).

2

Gated Fusion

A learnable sigmoid gate blends token embeddings with scalar projections: (E = E_{\text{tok}} + \sigma(g) \cdot E_{\text{val}}).

3

Per-Token MLPs

Each feature has its own MLP for scalar projection, enabling feature-specific transformations.

4

TokenDrop

Feature-level dropout that zeros entire embeddings during training ((p = 0.12)).

5

CLS Token

A learned [CLS] token aggregates global information for prediction.

6

Pre-LayerNorm

Layer normalization is applied before attention blocks for improved training stability.

🛠️ End-to-End Pipeline

The pipeline is leak-free, modular, and stacking-aware.

graph TD
    subgraph "1. Data Preparation"
        A[Raw Data] -->|K-Fold CV| B[Train Base Model (Ridge)]
        B --> C[Compute OOF Residuals]
    end

    subgraph "2. Tokenization"
        C --> D[TabularTokenizer]
        D -->|Quantile Binning| E[Discrete Tokens]
        D -->|Z-Score Scaling| F[Continuous Scalars]
    end

    subgraph "3. TabTransformer++"
        E & F --> G[Gated Fusion]
        G --> H[TokenDrop]
        H --> I[Transformer Encoder × 3]
        I --> J[Residual Prediction]
    end

    subgraph "4. Post-Processing"
        J --> K[Isotonic Calibration]
        K --> L[Add to Base Prediction]
    end

⚙️ Repository Components

Config

Centralized configuration for:

Batch size

Learning rate

Number of quantile bins

Hidden dimensions

TokenDrop probability

Training and model hyperparameters

get_simulated_data()

Simulates a stacking environment by:

Training Ridge and RandomForest models

Using K-Fold Cross-Validation

Producing leak-free Out-of-Fold (OOF) residuals

TabularTokenizer

Custom tokenizer that:

Applies quantile binning for discrete tokens

Applies z-score normalization for continuous scalars

Outputs both representations per feature

TabTransformerGated

Main PyTorch model implementing:

Dual token + scalar inputs

Gated fusion mechanism

Per-feature scalar MLPs

[CLS] token aggregation

Transformer encoder stack with Pre-LayerNorm

TokenDrop

Feature-level dropout module that:

Randomly removes entire feature embeddings

Reduces reliance on any single column

Training Loop

Includes:

Supervised training on residual targets

EMA (Exponential Moving Average) of model weights

Isotonic Regression for final calibration

🔮 Final Inference

[ y_{\text{pred}} = y_{\text{base}} + y_{\text{residual}} ]

📊 Results & Performance

Residual stacking yields a substantial error reduction compared to the base model alone.

California Housing Simulation (Example)

Model Strategy

Train RMSE (CV)

Test RMSE (Holdout)

Improvement

Base Model (Ridge)

0.8094

0.7361

—

✅ Key Takeaways

Residual learning simplifies the learning task for deep models.

TabTransformer++ effectively models continuous features without losing precision.

Stacking linear and Transformer models combines interpretability with expressive power.

Leak-free OOF residuals are critical for reliable performance gains.


