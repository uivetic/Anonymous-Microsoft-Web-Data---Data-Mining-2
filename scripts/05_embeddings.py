import _bootstrap  # noqa: F401

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from msweb.config import FEATURE_SETS_DIR, FIGURES_DIR, PROCESSED_DIR, ensure_dirs
    from msweb.embeddings import coords_frame, run_tsne, svd_coords
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "zatim: python scripts/05_embeddings.py"
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


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)
    print(f"  {name}")


def plot_2d(frame: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(
        frame[x],
        frame[y],
        c=frame["n_visits"],
        s=4,
        cmap="viridis",
        alpha=0.55,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="broj poseta")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    save(fig, filename)


def plot_3d(frame: pd.DataFrame, cols: list[str], title: str, filename: str) -> None:
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        frame[cols[0]],
        frame[cols[1]],
        frame[cols[2]],
        c=frame["n_visits"],
        s=3,
        cmap="viridis",
        alpha=0.55,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="broj poseta")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.set_title(title)
    save(fig, filename)


def main() -> None:
    ensure_dirs()
    users = pd.read_csv(PROCESSED_DIR / "users_train.csv")
    A5 = np.load(FEATURE_SETS_DIR / "A5" / "X.npy")

    print(f"svi korisnici: {len(users)}  A5 oblik: {A5.shape}")

    # SVD koordinate — linearna projekcija, jeftina.
    svd3 = svd_coords(A5, 3)
    svd_frame = coords_frame(users, svd3, "svd")
    svd_frame.to_csv(PROCESSED_DIR / "coords_svd.csv", index=False)

    # t-SNE nad A5 (euklidsko). Pun Jaccard 22k×22k bi bio ~2GB i
    # zahtevao exact t-SNE; A5 vec nosi ~90% varijanse binarne matrice.
    print("t-SNE 2D nad A5...")
    tsne2 = run_tsne(A5, 2, metric="euclidean")
    print("t-SNE 3D nad A5...")
    tsne3 = run_tsne(A5, 3, metric="euclidean")

    tsne_frame = coords_frame(users, tsne2, "tsne2")
    for i in range(3):
        tsne_frame[f"tsne3_{i + 1}"] = tsne3[:, i]
    tsne_frame.to_csv(PROCESSED_DIR / "coords_tsne.csv", index=False)

    print("grafici:")
    plot_2d(
        svd_frame,
        "svd_1",
        "svd_2",
        "SVD (A5) — prve dve komponente",
        "embedding_svd_2d.png",
    )
    plot_3d(
        svd_frame,
        ["svd_1", "svd_2", "svd_3"],
        "SVD (A5) — prve tri komponente",
        "embedding_svd_3d.png",
    )
    plot_2d(
        tsne_frame,
        "tsne2_1",
        "tsne2_2",
        "t-SNE nad A5 ugradnjom (svi korisnici)",
        "embedding_tsne_2d.png",
    )
    plot_3d(
        tsne_frame,
        ["tsne3_1", "tsne3_2", "tsne3_3"],
        "t-SNE 3D nad A5 ugradnjom (svi korisnici)",
        "embedding_tsne_3d.png",
    )

    print(f"\nzapisano u {PROCESSED_DIR} i {FIGURES_DIR}")


if __name__ == "__main__":
    main()
