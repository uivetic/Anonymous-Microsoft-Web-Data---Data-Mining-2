import _bootstrap  # noqa: F401

import sys

import numpy as np
import pandas as pd

try:
    from msweb.algorithms.clarans import Clarans
    from msweb.config import (
        CLARANS_K,
        CLARANS_MAX_ATTEMPTS,
        CLARANS_NUM_LOCAL,
        FEATURE_SET_NAMES,
        FEATURE_SETS_DIR,
        RANDOM_STATE,
        TABLES_DIR,
        ensure_dirs,
    )
    from msweb.data_io import load_feature_matrix, load_users
    from msweb.evaluation import append_result, evaluate_clustering, save_labels, save_model
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "  source .venv/bin/activate\n"
        "zatim: python scripts/08_clarans.py"
    )


def _metric_for(meta: dict) -> str:
    return "jaccard" if meta.get("kind") == "columns" else "euclidean"


def _medoid_vroots(X_row: np.ndarray, columns: pd.DataFrame) -> str:
    visited = np.flatnonzero(X_row > 0)
    if len(visited) == 0:
        return ""
    titles = columns.iloc[visited]["title"].astype(str).tolist()
    return "; ".join(titles)


def build_medoid_rows(
    *,
    feature_set: str,
    users: pd.DataFrame,
    X: np.ndarray,
    meta: dict,
    medoids: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    rows = []
    columns = None
    if meta.get("kind") == "columns":
        columns = pd.read_csv(FEATURE_SETS_DIR / feature_set / "columns.csv")

    for slot, row_idx in enumerate(medoids):
        row_idx = int(row_idx)
        size = int(np.sum(labels == slot))
        user = users.iloc[row_idx]["user"]
        entry = {
            "feature_set": feature_set,
            "medoid_slot": slot,
            "row": row_idx,
            "user": user,
            "cluster_size": size,
            "n_visits_in_set": int(np.sum(X[row_idx] > 0)) if meta.get("kind") == "columns" else "",
            "vroots": "",
        }
        if columns is not None:
            entry["vroots"] = _medoid_vroots(X[row_idx], columns)
        rows.append(entry)
    return rows


def main() -> None:
    ensure_dirs()
    users = load_users()
    summaries = []
    medoid_rows: list[dict] = []

    print(
        f"CLARANS: k={CLARANS_K}, num_local={CLARANS_NUM_LOCAL}, "
        f"max_attempts={CLARANS_MAX_ATTEMPTS}, "
        f"skupovi={list(FEATURE_SET_NAMES)}"
    )

    for name in FEATURE_SET_NAMES:
        print(f"\n=== {name} ===")
        X, meta = load_feature_matrix(name)
        metric = _metric_for(meta)
        model = Clarans(
            k=CLARANS_K,
            max_attempts=CLARANS_MAX_ATTEMPTS,
            num_local=CLARANS_NUM_LOCAL,
            metric=metric,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        assert model.medoids_ is not None and model.cost_ is not None

        metrics = evaluate_clustering(X, labels, metric=metric)
        append_result(
            {
                "algorithm": "clarans",
                "feature_set": name,
                "k": CLARANS_K,
                "distance": metric,
                **metrics,
                "params": {
                    "k": CLARANS_K,
                    "num_local": CLARANS_NUM_LOCAL,
                    "max_attempts": CLARANS_MAX_ATTEMPTS,
                    "metric": metric,
                    "cost": model.cost_,
                    "kind": meta["kind"],
                },
            }
        )
        stem = f"clarans_{name}_k{CLARANS_K}"
        save_model(
            {
                "model": model,
                "feature_set": name,
                "k": CLARANS_K,
                "medoids": model.medoids_,
                "cost": model.cost_,
                "metric": metric,
            },
            stem,
        )
        save_labels(users, labels, stem)
        medoid_rows.extend(
            build_medoid_rows(
                feature_set=name,
                users=users,
                X=X,
                meta=meta,
                medoids=model.medoids_,
                labels=labels,
            )
        )

        summaries.append(
            {
                "feature_set": name,
                "k": CLARANS_K,
                "distance": metric,
                "cost": model.cost_,
                "silhouette": metrics["silhouette"],
                "intra_inter": metrics["intra_inter"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
                "n_features": meta["n_features"],
            }
        )
        print(
            f"  metric={metric}  cost={model.cost_:.1f}  "
            f"sil={metrics['silhouette']:.4f}  "
            f"intra/inter={metrics['intra_inter']:.3f}"
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES_DIR / "clarans_summary.csv", index=False)
    pd.DataFrame(medoid_rows).to_csv(TABLES_DIR / "clarans_medoids.csv", index=False)
    print("\n=== rezime ===")
    print(summary.to_string(index=False))
    print(f"\nmedoids -> {TABLES_DIR / 'clarans_medoids.csv'}")


if __name__ == "__main__":
    main()
