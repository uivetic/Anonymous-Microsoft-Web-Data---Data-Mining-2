"""Brza provera modula evaluacije na A2 + KMeans (nije deo finalnih eksperimenata)."""

import _bootstrap  # noqa: F401

import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans

try:
    from msweb.config import FEATURE_SETS_DIR, PROCESSED_DIR, RANDOM_STATE, ensure_dirs
    from msweb.evaluation import (
        append_result,
        evaluate_clustering,
        save_labels,
        save_model,
    )
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "  source .venv/bin/activate\n"
        "zatim: python scripts/06_eval_smoke.py"
    )


def main() -> None:
    ensure_dirs()
    users = pd.read_csv(PROCESSED_DIR / "users_train.csv")
    X = sparse.load_npz(FEATURE_SETS_DIR / "A2" / "X.npz").toarray().astype(np.float64)

    model = KMeans(n_clusters=6, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)

    eucl = evaluate_clustering(X, labels, metric="euclidean")
    jacc = evaluate_clustering(X.astype(bool), labels, metric="jaccard")

    print("euclidean:", {k: round(v, 4) if isinstance(v, float) else v for k, v in eucl.items()})
    print("jaccard:  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in jacc.items()})

    append_result(
        {
            "algorithm": "kmeans_smoke",
            "feature_set": "A2",
            "k": 6,
            "distance": "euclidean",
            **eucl,
            "params": {"n_init": 10, "note": "smoke test"},
        }
    )
    save_model(
        {"model": model, "feature_set": "A2", "labels": labels},
        "kmeans_smoke_A2_k6",
    )
    save_labels(users, labels, "kmeans_smoke_A2_k6")
    print("registar i model snimljeni (smoke)")


if __name__ == "__main__":
    main()
