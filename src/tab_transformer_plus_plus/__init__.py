"""
TabTransformer++ package exposing the key classes and functions.

The package includes:

- **TabTransformerGated**: transformer model with dual representation and gated fusion
- **TabularTokenizer**: quantile binning and robust scaler for dual representation
- **train_pipeline**: high‑level function to train the model on a dataset

Usage:

```python
from tab_transformer_plus_plus import TabTransformerGated, TabularTokenizer

tokenizer = TabularTokenizer(n_bins=32)
tokenizer.fit(df)
```
"""

from .model import TabTransformerGated, PerTokenValMLP, TokenDrop
from .tokenizer import TabularTokenizer
from .utils import seed_everything, root_rmse

__all__ = [
    "TabTransformerGated",
    "PerTokenValMLP",
    "TokenDrop",
    "TabularTokenizer",
    "seed_everything",
    "root_rmse",
]