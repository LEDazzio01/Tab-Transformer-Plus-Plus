# TabTransformer++ for Residual Learning

This notebook implements **TabTransformer++**, an enhanced deep learning architecture for tabular data.

Instead of predicting the target variable directly, this model demonstrates a **Residual Learning (Model Stacking)** strategy: a simple linear model makes a baseline prediction, and the Transformer is trained to predict the **error (residual)** of that baseline. The final model combines both and significantly outperforms either model alone.

Based on: **TabTransformer: Tabular Data Modeling Using Contextual Embeddings** (Huang et al., 2020)

---

## 🧠 Core concept: Residual stacking

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

This design matches how human analysts often work: start with a simple baseline, then layer in structured corrections.

---

## 🚀 Architectural innovations

Standard Transformers treat tabular data as simple discrete tokens (like words). **TabTransformer++** introduces six key innovations to handle **continuous numerical data** more effectively.

### Innovation summary

| # | Innovation        | Description |
|---|-------------------|-------------|
| 1 | Dual Representation | Features are processed as both discrete tokens (via Quantile Binning) and continuous scalars (via Z-Score normalization) to capture both non-linear patterns and precise magnitudes. |
| 2 | Gated Fusion      | A learnable Sigmoid Gate dynamically blends the discrete embedding with the continuous scalar projection: \(E = E_{\text{tok}} + \sigma(g) \cdot E_{\text{val}}\). |
| 3 | Per-Token MLPs    | Each feature has its own dedicated MLP to project scalar values, allowing the model to learn specific transformations (e.g., logarithmic vs. linear) for each column. |
| 4 | TokenDrop         | "Dropout at the feature level." Randomly zeros out entire feature embeddings during training (\(p = 0.12\)) to prevent over-reliance on any single column. |
| 5 | CLS Token         | A special learned `[CLS]` token is prepended to the input sequence. The final prediction is derived solely from this token's embedding. |
| 6 | Pre-LayerNorm     | Layer Normalization is applied *before* attention blocks (`norm_first=True`) for stable gradient flow. |

---

## 🛠️ Workflow diagram

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

