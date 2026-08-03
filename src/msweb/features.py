from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from msweb.config import (
    A1_MIN_VROOT_USERS,
    A2_TOP_N,
    A3_MIN_VARIANCE,
    A4_MAX_PHI,
    A5_MAX_COMPONENTS,
    A5_VARIANCE_RATIO,
    RANDOM_STATE,
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    kind: str  # "columns" | "embedding"
    description: str
    columns: np.ndarray | None  # indeksi kolona u X_train
    matrix: sparse.spmatrix | np.ndarray
    meta: dict


def column_stats(X: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    X = X.tocsr()
    counts = np.asarray(X.sum(axis=0)).ravel().astype(float)
    n_users = X.shape[0]
    p = counts / n_users
    variance = p * (1.0 - p)
    return counts, variance


def select_all(n_cols: int) -> np.ndarray:
    return np.arange(n_cols, dtype=np.int32)


def select_by_min_users(counts: np.ndarray, min_users: int) -> np.ndarray:
    return np.flatnonzero(counts >= min_users).astype(np.int32)


def select_top_n(counts: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(counts)[::-1]
    return order[:n].astype(np.int32)


def select_by_variance(variance: np.ndarray, min_variance: float) -> np.ndarray:
    return np.flatnonzero(variance >= min_variance).astype(np.int32)


def select_nonredundant(
    X: sparse.spmatrix,
    candidate_cols: np.ndarray,
    variance: np.ndarray,
    max_phi: float,
) -> np.ndarray:
    """Greedy: zadrzi kolone sa vecom varijansom ako |phi| sa vec zadrzanim < max_phi."""
    if len(candidate_cols) == 0:
        return candidate_cols

    order = candidate_cols[np.argsort(variance[candidate_cols])[::-1]]
    dense = X[:, order].toarray().astype(np.float64)
    # phi za binarne == Pearsonova korelacija
    corr = np.corrcoef(dense, rowvar=False)
    np.fill_diagonal(corr, 0.0)

    keep_local: list[int] = []
    for i in range(len(order)):
        if not keep_local:
            keep_local.append(i)
            continue
        if np.max(np.abs(corr[i, keep_local])) < max_phi:
            keep_local.append(i)

    return order[np.array(keep_local, dtype=np.int32)]


def choose_n_components(
    explained_variance_ratio: np.ndarray, target: float
) -> int:
    cumulative = np.cumsum(explained_variance_ratio)
    return int(min(len(cumulative), np.searchsorted(cumulative, target) + 1))


def fit_svd(
    X: sparse.spmatrix,
    variance_ratio: float = A5_VARIANCE_RATIO,
    max_components: int = A5_MAX_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, TruncatedSVD]:
    # TruncatedSVD ne prima ciljni udeo varijanse, pa prvo probamo gornju
    # granicu komponenti, zatim refitujemo sa najmanjim k koje dize prag.
    max_k = min(max_components, X.shape[1] - 1, X.shape[0] - 1)
    probe = TruncatedSVD(n_components=max_k, random_state=random_state)
    probe.fit(X)
    n_components = choose_n_components(probe.explained_variance_ratio_, variance_ratio)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(X)
    return embedding.astype(np.float64), svd


def build_feature_sets(
    X: sparse.csr_matrix,
    features: pd.DataFrame,
    *,
    a1_min_users: int = A1_MIN_VROOT_USERS,
    a2_top_n: int = A2_TOP_N,
    a3_min_variance: float = A3_MIN_VARIANCE,
    a4_max_phi: float = A4_MAX_PHI,
    a5_variance_ratio: float = A5_VARIANCE_RATIO,
    a5_max_components: int = A5_MAX_COMPONENTS,
) -> tuple[list[FeatureSet], TruncatedSVD]:
    counts, variance = column_stats(X)
    n_cols = X.shape[1]

    a0_cols = select_all(n_cols)
    a1_cols = select_by_min_users(counts, a1_min_users)
    a2_cols = select_top_n(counts, a2_top_n)
    a3_cols = select_by_variance(variance, a3_min_variance)
    a4_cols = select_nonredundant(X, a0_cols, variance, a4_max_phi)
    embedding, svd = fit_svd(
        X, variance_ratio=a5_variance_ratio, max_components=a5_max_components
    )

    def pack(
        name: str,
        cols: np.ndarray,
        description: str,
        extra: dict | None = None,
    ) -> FeatureSet:
        meta = {
            "n_features": int(len(cols)),
            "n_users": int(X.shape[0]),
            "n_visits_covered": float(counts[cols].sum()),
            "visit_coverage": float(counts[cols].sum() / counts.sum()),
            "vroot_ids": features.loc[cols, "vroot_id"].astype(int).tolist(),
        }
        if extra:
            meta.update(extra)
        return FeatureSet(
            name=name,
            kind="columns",
            description=description,
            columns=cols,
            matrix=X[:, cols].tocsr(),
            meta=meta,
        )

    sets = [
        pack("A0", a0_cols, "sve oblasti (baseline)"),
        pack(
            "A1",
            a1_cols,
            f"oblasti sa >= {a1_min_users} korisnika",
            {"min_users": a1_min_users},
        ),
        pack(
            "A2",
            a2_cols,
            f"top {a2_top_n} najposecenijih oblasti",
            {"top_n": a2_top_n},
        ),
        pack(
            "A3",
            a3_cols,
            f"binarna varijansa p(1-p) >= {a3_min_variance}",
            {"min_variance": a3_min_variance},
        ),
        pack(
            "A4",
            a4_cols,
            f"bez redundantnih parova (|phi| < {a4_max_phi})",
            {"max_phi": a4_max_phi},
        ),
        FeatureSet(
            name="A5",
            kind="embedding",
            description=(
                f"TruncatedSVD: najmanje komponenti za "
                f">= {a5_variance_ratio:.0%} objasnjene varijanse"
            ),
            columns=None,
            matrix=embedding,
            meta={
                "n_features": int(embedding.shape[1]),
                "n_users": int(X.shape[0]),
                "explained_variance_ratio_sum": float(
                    svd.explained_variance_ratio_.sum()
                ),
                "n_components": int(embedding.shape[1]),
                "target_variance_ratio": float(a5_variance_ratio),
                "max_components_probed": int(a5_max_components),
            },
        ),
    ]
    return sets, svd


def feature_sets_summary(sets: list[FeatureSet]) -> pd.DataFrame:
    rows = []
    for fs in sets:
        rows.append(
            {
                "set": fs.name,
                "kind": fs.kind,
                "description": fs.description,
                "n_features": fs.meta["n_features"],
                "n_users": fs.meta["n_users"],
                "visit_coverage": fs.meta.get("visit_coverage"),
                "explained_variance_ratio_sum": fs.meta.get(
                    "explained_variance_ratio_sum"
                ),
            }
        )
    return pd.DataFrame(rows)
