import numpy as np
import pandas as pd
from scipy import sparse

from msweb.config import MIN_USER_VISITS, MODEL_SOURCE


def select_model_visits(
    users: pd.DataFrame,
    visits: pd.DataFrame,
    source: str = MODEL_SOURCE,
    min_user_visits: int = MIN_USER_VISITS,
) -> pd.DataFrame:
    model_users = users.loc[users["source"] == source, "user"]
    selected = visits[visits["user"].isin(model_users)]

    per_user = selected.groupby("user").size()
    keep_users = per_user[per_user >= min_user_visits].index
    return selected[selected["user"].isin(keep_users)].copy()


def build_binary_matrix(
    visits: pd.DataFrame, vroots: pd.DataFrame
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    user_index = (
        visits.groupby("user")
        .size()
        .rename("n_visits")
        .reset_index()
        .sort_values("user")
        .reset_index(drop=True)
    )
    user_index["row"] = np.arange(len(user_index), dtype=np.int32)

    feature_index = (
        visits.groupby("vroot_id")
        .size()
        .rename("n_visits")
        .reset_index()
        .merge(vroots, on="vroot_id", how="left")
        .sort_values("vroot_id")
        .reset_index(drop=True)
    )
    feature_index["col"] = np.arange(len(feature_index), dtype=np.int32)

    mapped = visits.merge(user_index[["user", "row"]], on="user").merge(
        feature_index[["vroot_id", "col"]], on="vroot_id"
    )

    matrix = sparse.csr_matrix(
        (
            np.ones(len(mapped), dtype=np.uint8),
            (mapped["row"].to_numpy(), mapped["col"].to_numpy()),
        ),
        shape=(len(user_index), len(feature_index)),
    )

    # Posete su jedinstvene (user, vroot); ovo hvata grešku u pipeline-u.
    if matrix.nnz != len(visits):
        raise ValueError(
            f"ocekivano {len(visits)} jedinica u matrici, dobijeno {matrix.nnz}"
        )

    return matrix, user_index, feature_index[
        ["col", "vroot_id", "title", "url", "n_visits"]
    ]


def preprocess_summary(
    users: pd.DataFrame,
    visits: pd.DataFrame,
    selected: pd.DataFrame,
    matrix: sparse.csr_matrix,
    source: str = MODEL_SOURCE,
    min_user_visits: int = MIN_USER_VISITS,
) -> pd.DataFrame:
    before_users = users.loc[users["source"] == source, "user"]
    before_visits = visits[visits["user"].isin(before_users)]
    return pd.DataFrame(
        [
            {
                "source": source,
                "min_user_visits": min_user_visits,
                "users_before": before_users.nunique(),
                "users_after": selected["user"].nunique(),
                "visits_before": len(before_visits),
                "visits_after": len(selected),
                "vroots_before": before_visits["vroot_id"].nunique(),
                "vroots_after": selected["vroot_id"].nunique(),
                "matrix_rows": matrix.shape[0],
                "matrix_cols": matrix.shape[1],
                "matrix_nnz": matrix.nnz,
                "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
            }
        ]
    )
