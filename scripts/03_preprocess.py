import _bootstrap  # noqa: F401

import json
import sys

import pandas as pd
from scipy import sparse

try:
    from msweb.config import INTERIM_DIR, PROCESSED_DIR, TABLES_DIR, ensure_dirs
    from msweb.preprocess import (
        build_binary_matrix,
        preprocess_summary,
        select_model_visits,
    )
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "Iz korena projekta pokreni:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "zatim: python scripts/03_preprocess.py"
    )


def main() -> None:
    ensure_dirs()
    vroots = pd.read_csv(INTERIM_DIR / "vroots.csv")
    users = pd.read_csv(INTERIM_DIR / "users.csv")
    visits = pd.read_csv(INTERIM_DIR / "visits.csv")

    selected = select_model_visits(users, visits)
    matrix, user_index, feature_index = build_binary_matrix(selected, vroots)
    summary = preprocess_summary(users, visits, selected, matrix)

    sparse.save_npz(PROCESSED_DIR / "X_train.npz", matrix)
    user_index.to_csv(PROCESSED_DIR / "users_train.csv", index=False)
    feature_index.to_csv(PROCESSED_DIR / "features.csv", index=False)
    selected.to_csv(PROCESSED_DIR / "visits_train.csv", index=False)
    summary.to_csv(TABLES_DIR / "preprocess_summary.csv", index=False)

    meta = {
        "matrix": "X_train.npz",
        "users": "users_train.csv",
        "features": "features.csv",
        "visits": "visits_train.csv",
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "format": "csr",
        "values": "binary",
    }
    (PROCESSED_DIR / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    row = summary.iloc[0]
    print(
        f"source={row['source']}  min_user_visits={row['min_user_visits']}\n"
        f"users: {row['users_before']} -> {row['users_after']}\n"
        f"vroots: {row['vroots_before']} -> {row['vroots_after']}\n"
        f"visits: {row['visits_before']} -> {row['visits_after']}\n"
        f"matrix: {matrix.shape[0]} x {matrix.shape[1]}  "
        f"nnz={matrix.nnz}  density={row['density']:.4%}\n"
        f"zapisano u {PROCESSED_DIR}"
    )


if __name__ == "__main__":
    main()
