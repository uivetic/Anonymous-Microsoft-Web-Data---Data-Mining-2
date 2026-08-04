from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from msweb.config import (
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    KMEANS_N_INIT,
    KMEANS_SELECTED_K,
    RANDOM_STATE,
)


@dataclass
class KMeansSweepResult:
    table: pd.DataFrame
    best_k: int
    best_labels: np.ndarray
    best_model: KMeans


def fit_kmeans(
    X: np.ndarray,
    k: int,
    *,
    n_init: int = KMEANS_N_INIT,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
        init="k-means++",
    )
    model.fit(X)
    return model


def sweep_kmeans(
    X: np.ndarray,
    *,
    k_min: int = KMEANS_K_MIN,
    k_max: int = KMEANS_K_MAX,
    selected_k: int = KMEANS_SELECTED_K,
    n_init: int = KMEANS_N_INIT,
    random_state: int = RANDOM_STATE,
    silhouette_sample_size: int = 5000,
) -> KMeansSweepResult:
    if not (k_min <= selected_k <= k_max):
        raise ValueError(f"selected_k={selected_k} van opsega [{k_min}, {k_max}]")

    rows = []
    models: dict[int, KMeans] = {}
    labels_by_k: dict[int, np.ndarray] = {}

    for k in range(k_min, k_max + 1):
        model = fit_kmeans(X, k, n_init=n_init, random_state=random_state)
        labels = model.labels_
        sil_kw = {"metric": "euclidean"}
        if len(X) > silhouette_sample_size:
            sil_kw["sample_size"] = silhouette_sample_size
            sil_kw["random_state"] = random_state
        silhouette = float(silhouette_score(X, labels, **sil_kw))
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
                "n_iter": int(model.n_iter_),
            }
        )
        models[k] = model
        labels_by_k[k] = labels

    table = pd.DataFrame(rows)
    # Izbor: ručni lakat (KMEANS_SELECTED_K); silueta ostaje u tabeli/grafiku.
    best_k = int(selected_k)
    return KMeansSweepResult(
        table=table,
        best_k=best_k,
        best_labels=labels_by_k[best_k],
        best_model=models[best_k],
    )
