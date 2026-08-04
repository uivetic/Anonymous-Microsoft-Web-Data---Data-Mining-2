from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)

from msweb.config import MODELS_DIR, RESULTS_REGISTRY, ensure_dirs

NOISE_LABEL = -1


def _finite_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    return labels, mask


def n_clusters(labels: np.ndarray) -> int:
    labels, mask = _finite_labels(labels)
    if not mask.any():
        return 0
    return int(len(np.unique(labels[mask])))


def n_noise(labels: np.ndarray) -> int:
    return int(np.sum(np.asarray(labels) == NOISE_LABEL))


def sse(X: np.ndarray, labels: np.ndarray) -> float:
    """Zbir kvadrata udaljenosti do centroida (euklidsko; ignoriše šum)."""
    labels, mask = _finite_labels(labels)
    if not mask.any():
        return float("nan")
    X = np.asarray(X)[mask]
    y = labels[mask]
    total = 0.0
    for lab in np.unique(y):
        block = X[y == lab]
        center = block.mean(axis=0)
        total += float(np.sum((block - center) ** 2))
    return total


def intra_inter_ratio(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
) -> float:
    """
    prosek unutar-klaster / prosek između-centroida.
    Manje je bolje (kompaktniji klasteri, dalji centri).
    """
    labels, mask = _finite_labels(labels)
    if mask.sum() < 2:
        return float("nan")
    X = np.asarray(X)[mask]
    y = labels[mask]
    if metric == "jaccard":
        X = X.astype(bool)
    labs = np.unique(y)
    if len(labs) < 2:
        return float("nan")

    centers = []
    intra_parts = []
    for lab in labs:
        block = X[y == lab]
        center = block.mean(axis=0)
        centers.append(center)
        if len(block) > 1:
            d = pairwise_distances(block, metric=metric)
            intra_parts.append(d[np.triu_indices(len(block), k=1)].mean())
        else:
            intra_parts.append(0.0)

    centers = np.asarray(centers)
    inter = pairwise_distances(centers, metric=metric)
    inter_mean = inter[np.triu_indices(len(centers), k=1)].mean()
    if inter_mean <= 0:
        return float("nan")
    return float(np.mean(intra_parts) / inter_mean)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Vrati metrike za jednu particiju.
    Silueta podržava metric='euclidean'|'jaccard'|'precomputed'.
    CH i DB u sklearn traže euklidski prostor atributa (ne precomputed).
    """
    labels = np.asarray(labels)
    X = np.asarray(X)
    _, mask = _finite_labels(labels)
    n_cl = n_clusters(labels)
    result = {
        "n_clusters": float(n_cl),
        "n_noise": float(n_noise(labels)),
        "n_clustered": float(mask.sum()),
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "sse": float("nan"),
        "intra_inter": float("nan"),
    }

    if n_cl < 2 or mask.sum() < n_cl + 1:
        return result

    Xc = X[mask]
    yc = labels[mask]
    if metric == "jaccard":
        Xc = Xc.astype(bool)

    try:
        # Za velike skupove silueta se uzorkuje (posebno skupo za jaccard).
        kw: dict[str, Any] = {"metric": metric}
        if sample_size is not None and len(Xc) > sample_size and metric != "precomputed":
            kw["sample_size"] = sample_size
            kw["random_state"] = random_state
        result["silhouette"] = float(silhouette_score(Xc, yc, **kw))
    except ValueError:
        pass

    if metric == "euclidean":
        try:
            result["calinski_harabasz"] = float(calinski_harabasz_score(Xc, yc))
        except ValueError:
            pass
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(Xc, yc))
        except ValueError:
            pass
        result["sse"] = sse(Xc, yc)
        result["intra_inter"] = intra_inter_ratio(Xc, yc, metric="euclidean")
    elif metric == "jaccard":
        # CH/DB/SSE nisu prirodni za Jaccard; intra/inter na uzorku.
        if sample_size is not None and len(Xc) > sample_size:
            rng = np.random.default_rng(random_state)
            take = rng.choice(len(Xc), size=sample_size, replace=False)
            result["intra_inter"] = intra_inter_ratio(
                Xc[take], yc[take], metric="jaccard"
            )
        else:
            result["intra_inter"] = intra_inter_ratio(Xc, yc, metric="jaccard")

    return result


def append_result(
    row: dict[str, Any],
    path: Path = RESULTS_REGISTRY,
) -> pd.DataFrame:
    ensure_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **row,
    }
    if "params" in record and not isinstance(record["params"], str):
        record["params"] = json.dumps(record["params"], sort_keys=True)

    frame = pd.DataFrame([record])
    if path.exists():
        prev = pd.read_csv(path)
        frame = pd.concat([prev, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def save_model(
    payload: dict[str, Any],
    name: str,
    directory: Path = MODELS_DIR,
) -> Path:
    ensure_dirs()
    path = directory / f"{name}.joblib"
    joblib.dump(payload, path)
    return path


def save_labels(
    users: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    directory: Path = MODELS_DIR,
) -> Path:
    ensure_dirs()
    path = directory / f"{name}_labels.csv"
    out = users[["user", "row"]].copy() if "row" in users.columns else users[["user"]].copy()
    out["cluster"] = np.asarray(labels)
    out.to_csv(path, index=False)
    return path
