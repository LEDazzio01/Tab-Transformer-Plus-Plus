# TabTransformer++ for Residual Learning

This notebook implements **TabTransformer++**, an enhanced deep learning architecture for tabular data.

Instead of predicting the target variable directly, this model demonstrates a **Residual Learning (Model Stacking)** strategy: a simple linear model makes a baseline prediction, and the Transformer is trained to predict the **error (residual)** of that baseline. The final model combines both and significantly outperforms either model alone.

Based on: **TabTransformer: Tabular Data Modeling Using Contextual Embeddings** (Huang et al., 2020)

---

## 🧠 Core Concept: Residual Stacking

The model treats the regression problem as an **additive decomposition**:



\[
\text{Final Prediction} = \text{Base Prediction} + \text{Predicted Residual}
\]



- **Base Model (Ridge):** A computationally cheap, interpretable model generates the initial prediction.
- **Target Calculation:**



\[
\text{Residual} = \text{True Target} - \text{Base Prediction}
\]



- **Correction:** The TabTransformer++ learns to predict this specific error.
- **Inference:** The Transformer's output is added to the Ridge prediction to correct it.

This design mirrors how human analysts often work: start with a simple baseline, then layer in structured corrections.

---

## 🚀 Architectural Innovations

Standard Transformers treat tabular data as simple discrete tokens (like words). **TabTransformer++** introduces six key innovations to handle **continuous numerical data** more effectively.

### Innovation Summary

| # | Innovation | Description |
|---|------------|-------------|
| 1 | Dual Representation | Features are processed as both discrete tokens (via Quantile Binning) and continuous scalars (via Z-Score normalization) to capture both non-linear patterns and precise magnitudes. |
| 2 | Gated Fusion | A learnable sigmoid gate dynamically blends the discrete embedding with the continuous scalar projection: \(E = E_{\text{tok}} + \sigma(g) \cdot E_{\text{val}}\). |
| 3 | Per-Token MLPs | Each feature has its own dedicated MLP to project scalar values, allowing the model to learn specific transformations (e.g., logarithmic vs. linear) for each column. |
| 4 | TokenDrop | Feature-level dropout that randomly zeros entire feature embeddings during training (\(p = 0.12\)) to prevent over-reliance on any single column. |
| 5 | CLS Token | A special learned `[CLS]` token is prepended to the input sequence. The final prediction is derived solely from this token’s embedding. |
| 6 | Pre-LayerNorm | Layer normalization is applied *before* attention blocks (`norm_first=True`) for stable gradient flow. |

---

## 🛠️ Workflow Diagram

The full pipeline is **leak-free**, modular, and stacking-aware.

```mermaid
graph TD
    subgraph "1. Data Prep (Leak-Free)"
        A[Raw Data] -->|K-Fold| B[Train Base Model: Ridge]
        B --> C[Calculate Residuals]
    end

    subgraph "2. Tokenization"
        C --> D{TabularTokenizer}
        D -->|Quantile Binning| E[Discrete Tokens]
        D -->|Z-Score Scaling| F[Continuous Scalars]
    end

    subgraph "3. TabTransformer++"
        E & F --> G[Gated Fusion Layer]
        G --> H[TokenDrop]
        H --> I[Transformer Encoder × 3]
        I --> J[Predict Residual]
    end

    subgraph "4. Post-Processing"
        J --> K[Isotonic Calibration]
        K --> L[Add to Base Prediction]
    end


## Config
Central repository for hyperparameters, including:
- Batch size
- Learning rate
- Number of bins
- Hidden sizes
- TokenDrop probability
- Other training and model parameters

## get_simulated_data()
Simulates the stacking environment by:
- Training **Ridge** and **RandomForest** models
- Using **K-Fold Cross-Validation**
- Generating leak-free **Out-Of-Fold (OOF) residuals**

## TabularTokenizer
Custom tokenizer that:
- Applies **Quantile Binning** to create discrete tokens
- Applies **Z-Score normalization** to create continuous scalars
- Outputs **both representations** for each feature

## TabTransformerGated (Main Model)
PyTorch module implementing:
- Dual representation inputs (tokens + scalars)
- Gated Fusion layer
- Per-token MLPs for scalar projections
- `[CLS]` token handling
- Transformer encoder stack with **Pre-LayerNorm**

## TokenDrop
Custom module for feature-level dropout:
- Randomly zeroes entire feature embeddings
- Controls over-reliance on any single column

## Training Loop
Implements:
- Standard supervised training on residual targets
- **EMA (Exponential Moving Average)** over model weights for stability
- **Isotonic Regression** for final value calibration

## Final Inference


\[
y_{\text{pred}} = y_{\text{base}} + y_{\text{residual}}
\]



---

# 📊 Results & Performance

The stacking approach yields a significant reduction in error compared to the base model alone.

## California Housing Simulation (Example)

| Model Strategy                 | Train RMSE (CV) | Test RMSE (Holdout) | Improvement |
|--------------------------------|-----------------|---------------------|-------------|
| Base Model Only (Ridge)        | 0.8094          | 0.7361             
