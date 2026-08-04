import _bootstrap  # noqa: F401

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from msweb.algorithms.kmeans import fit_kmeans, sweep_kmeans
    from msweb.config import (
        FEATURE_SET_NAMES,
        FIGURES_DIR,
        KMEANS_K_MAX,
        KMEANS_K_MIN,
        KMEANS_N_INIT,
        KMEANS_SELECTED_K,
        TABLES_DIR,
        ensure_dirs,
    )
    from msweb.data_io import load_feature_matrix, load_users
    from msweb.evaluation import append_result, evaluate_clustering, save_labels, save_model
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "  source .venv/bin/activate\n"
        "zatim: python scripts/07_kmeans.py"
    )

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_elbow(table: pd.DataFrame, feature_set: str, best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(table["k"], table["inertia"], marker="o")
    ax.axvline(best_k, color="crimson", linestyle="--", label=f"izabrano k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("SSE (inertia)")
    ax.set_title(f"Lakat — {feature_set}")
    ax.legend()

    ax = axes[1]
    ax.plot(table["k"], table["silhouette"], marker="o", color="#55A868")
    ax.axvline(best_k, color="crimson", linestyle="--", label=f"izabrano k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("silueta")
    ax.set_title(f"Silueta — {feature_set}")
    ax.legend()

    path = FIGURES_DIR / f"kmeans_elbow_{feature_set}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


def finalize(
    name: str,
    X,
    meta: dict,
    users: pd.DataFrame,
    table: pd.DataFrame,
    best_k: int,
    best_model,
    best_labels,
) -> dict:
    plot_elbow(table, name, best_k)
    metrics = evaluate_clustering(X, best_labels, metric="euclidean")
    append_result(
        {
            "algorithm": "kmeans",
            "feature_set": name,
            "k": best_k,
            "distance": "euclidean",
            **metrics,
            "params": {
                "k_min": KMEANS_K_MIN,
                "k_max": KMEANS_K_MAX,
                "selected_k": KMEANS_SELECTED_K,
                "n_init": KMEANS_N_INIT,
                "selection": "elbow_manual",
                "kind": meta["kind"],
            },
        }
    )
    stem = f"kmeans_{name}_k{best_k}"
    save_model(
        {
            "model": best_model,
            "feature_set": name,
            "best_k": best_k,
            "sweep": table.to_dict(orient="list"),
        },
        stem,
    )
    save_labels(users, best_labels, stem)
    print(
        f"  best_k={best_k}  "
        f"sil={metrics['silhouette']:.4f}  "
        f"CH={metrics['calinski_harabasz']:.1f}  "
        f"DB={metrics['davies_bouldin']:.3f}"
    )
    return {
        "feature_set": name,
        "best_k": best_k,
        "silhouette": metrics["silhouette"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "davies_bouldin": metrics["davies_bouldin"],
        "sse": metrics["sse"],
        "n_features": meta["n_features"],
    }


def main() -> None:
    ensure_dirs()
    users = load_users()
    reuse_sweep = "--reuse-sweep" in sys.argv
    summaries = []

    print(
        f"KMeans: sweep k={KMEANS_K_MIN}..{KMEANS_K_MAX}, "
        f"izabrano k={KMEANS_SELECTED_K} (lakat), "
        f"n_init={KMEANS_N_INIT}, skupovi={list(FEATURE_SET_NAMES)}"
        + (" [reuse-sweep]" if reuse_sweep else "")
    )

    for name in FEATURE_SET_NAMES:
        print(f"\n=== {name} ===")
        X, meta = load_feature_matrix(name)
        sweep_path = TABLES_DIR / f"kmeans_sweep_{name}.csv"

        if reuse_sweep and sweep_path.exists():
            table = pd.read_csv(sweep_path)
            if "feature_set" not in table.columns:
                table.insert(0, "feature_set", name)
            if KMEANS_SELECTED_K not in set(table["k"].astype(int)):
                sys.exit(f"U {sweep_path.name} nema k={KMEANS_SELECTED_K}")
            model = fit_kmeans(X, KMEANS_SELECTED_K)
            labels = model.labels_
            row = finalize(
                name, X, meta, users, table, KMEANS_SELECTED_K, model, labels
            )
        else:
            sweep = sweep_kmeans(X)
            sweep.table.insert(0, "feature_set", name)
            sweep.table.to_csv(sweep_path, index=False)
            row = finalize(
                name,
                X,
                meta,
                users,
                sweep.table,
                sweep.best_k,
                sweep.best_model,
                sweep.best_labels,
            )
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES_DIR / "kmeans_summary.csv", index=False)
    print("\n=== rezime ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
