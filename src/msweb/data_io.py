from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from msweb.config import FEATURE_SETS_DIR, PROCESSED_DIR


def load_users() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "users_train.csv")


def load_feature_matrix(name: str) -> tuple[np.ndarray, dict]:
    folder = FEATURE_SETS_DIR / name
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["kind"] == "embedding":
        X = np.load(folder / "X.npy")
    else:
        X = sparse.load_npz(folder / "X.npz").toarray().astype(np.float64)
    return X, meta
