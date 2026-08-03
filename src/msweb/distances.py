from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances


def jaccard_distance_matrix(X: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """Parovi Žakarovih rastojanja (1 - similarity) za binarne redove."""
    if sparse.issparse(X):
        dense = X.toarray()
    else:
        dense = np.asarray(X)
    # float32 stedi memoriju; sklearn jaccard ocekuje {0,1} vrednosti.
    return pairwise_distances(dense.astype(bool), metric="jaccard").astype(np.float32)
