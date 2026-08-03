from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from msweb.config import RANDOM_STATE, TSNE_PERPLEXITY


def svd_coords(embedding: np.ndarray, n_dims: int = 3) -> np.ndarray:
    return embedding[:, :n_dims].astype(np.float64)


def run_tsne(
    inputs: np.ndarray,
    n_components: int,
    *,
    metric: str = "euclidean",
    perplexity: float = TSNE_PERPLEXITY,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    init = "random" if metric == "precomputed" else "pca"
    perplexity = min(perplexity, max(5.0, (len(inputs) - 1) / 3))
    model = TSNE(
        n_components=n_components,
        metric=metric,
        perplexity=perplexity,
        init=init,
        random_state=random_state,
    )
    return model.fit_transform(inputs).astype(np.float64)


def coords_frame(
    users: pd.DataFrame, coords: np.ndarray, prefix: str
) -> pd.DataFrame:
    cols = {f"{prefix}_{i + 1}": coords[:, i] for i in range(coords.shape[1])}
    return pd.concat([users.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
