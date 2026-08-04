from __future__ import annotations

import numpy as np

from msweb.config import (
    CLARANS_K,
    CLARANS_MAX_ATTEMPTS,
    CLARANS_NUM_LOCAL,
    RANDOM_STATE,
)


class Clarans:

    def __init__(
        self,
        k: int = CLARANS_K,
        *,
        max_attempts: int = CLARANS_MAX_ATTEMPTS,
        num_local: int = CLARANS_NUM_LOCAL,
        metric: str = "jaccard",
        random_state: int = RANDOM_STATE,
    ) -> None:
        if k < 2:
            raise ValueError("k mora biti >= 2")
        if metric not in {"jaccard", "euclidean"}:
            raise ValueError(f"Nepoznata metrika: {metric}")
        self.k = k
        self.max_attempts = max_attempts
        self.num_local = num_local
        self.metric = metric
        self.random_state = random_state

        self.medoids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.cost_: float | None = None

    def fit(self, X: np.ndarray) -> Clarans:
        X = np.asarray(X)
        n = X.shape[0]
        if n < self.k:
            raise ValueError(f"n={n} < k={self.k}")

        if self.metric == "jaccard":
            Xb = X.astype(bool, copy=False)
            row_sums = Xb.sum(axis=1).astype(np.float64)
            state = {"X": Xb, "row_sums": row_sums}
            dist_col = _jaccard_column
        else:
            Xf = np.asarray(X, dtype=np.float64)
            state = {"X": Xf}
            dist_col = _euclidean_column

        rng = np.random.default_rng(self.random_state)
        best_medoids: np.ndarray | None = None
        best_cost = np.inf
        best_D: np.ndarray | None = None

        for _ in range(self.num_local):
            medoids = rng.choice(n, size=self.k, replace=False)
            D = np.column_stack([dist_col(state, m) for m in medoids])
            cost = float(D.min(axis=1).sum())

            attempts = 0
            while attempts < self.max_attempts:
                m_slot = int(rng.integers(0, self.k))
                non_medoid = _sample_non_medoid(rng, n, medoids)
                new_col = dist_col(state, non_medoid)
                D_trial = D.copy()
                D_trial[:, m_slot] = new_col
                new_cost = float(D_trial.min(axis=1).sum())
                if new_cost < cost:
                    medoids = medoids.copy()
                    medoids[m_slot] = non_medoid
                    D = D_trial
                    cost = new_cost
                    attempts = 0
                else:
                    attempts += 1

            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids.copy()
                best_D = D

        assert best_medoids is not None and best_D is not None
        self.medoids_ = best_medoids
        self.labels_ = best_D.argmin(axis=1).astype(np.int64)
        self.cost_ = best_cost
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_


def _sample_non_medoid(rng: np.random.Generator, n: int, medoids: np.ndarray) -> int:
    medoid_set = set(int(m) for m in medoids)
    while True:
        cand = int(rng.integers(0, n))
        if cand not in medoid_set:
            return cand


def _jaccard_column(state: dict, idx: int) -> np.ndarray:
    X: np.ndarray = state["X"]
    row_sums: np.ndarray = state["row_sums"]
    medoid = X[idx]
    medoid_sum = float(medoid.sum())
    inter = X @ medoid.astype(np.float64)
    union = row_sums + medoid_sum - inter
    dist = np.zeros(X.shape[0], dtype=np.float64)
    nz = union > 0
    dist[nz] = 1.0 - (inter[nz] / union[nz])
    return dist


def _euclidean_column(state: dict, idx: int) -> np.ndarray:
    X: np.ndarray = state["X"]
    diff = X - X[idx]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))
