import _bootstrap  # noqa: F401

import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from msweb.config import (
        FEATURE_SETS_DIR,
        PROCESSED_DIR,
        TABLES_DIR,
        ensure_dirs,
    )
    from msweb.features import build_feature_sets, feature_sets_summary
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "zatim: python scripts/04_features.py"
    )


def main() -> None:
    ensure_dirs()
    X = sparse.load_npz(PROCESSED_DIR / "X_train.npz").tocsr()
    features = pd.read_csv(PROCESSED_DIR / "features.csv")

    sets, svd = build_feature_sets(X, features)
    summary = feature_sets_summary(sets)

    for fs in sets:
        out_dir = FEATURE_SETS_DIR / fs.name
        out_dir.mkdir(parents=True, exist_ok=True)

        if fs.kind == "columns":
            sparse.save_npz(out_dir / "X.npz", fs.matrix)
            cols = features.loc[fs.columns].copy()
            cols.insert(0, "col_in_X_train", fs.columns)
            cols.to_csv(out_dir / "columns.csv", index=False)
        else:
            np.save(out_dir / "X.npy", fs.matrix)
            joblib.dump(svd, out_dir / "svd.joblib")
            components = pd.DataFrame(
                {
                    "component": np.arange(fs.matrix.shape[1]),
                    "explained_variance_ratio": svd.explained_variance_ratio_,
                }
            )
            components.to_csv(out_dir / "components.csv", index=False)

        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "name": fs.name,
                    "kind": fs.kind,
                    "description": fs.description,
                    **{
                        k: v
                        for k, v in fs.meta.items()
                        if k != "vroot_ids"  # vec u columns.csv
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary.to_csv(TABLES_DIR / "feature_sets_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nzapisano u {FEATURE_SETS_DIR}")


if __name__ == "__main__":
    main()
