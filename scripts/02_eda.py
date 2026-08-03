import _bootstrap  # noqa: F401

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from msweb.config import FIGURES_DIR, INTERIM_DIR, TABLES_DIR, ensure_dirs
    from msweb.eda import (
        basic_stats,
        distribution_summary,
        threshold_grid,
        vroot_jaccard,
        vroot_popularity,
    )
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "Iz korena projekta pokreni:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "zatim: python scripts/02_eda.py"
    )

VROOT_OPTIONS = [1, 5, 10, 20, 30, 50, 100, 200]
USER_OPTIONS = [1, 2, 3, 5]
HEATMAP_TOP_N = 25

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


def plot_visits_per_user(per_user: pd.Series) -> None:
    counts = per_user.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.to_numpy(), color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("broj posećenih oblasti")
    ax.set_ylabel("broj korisnika (log)")
    ax.set_title("Koliko oblasti sajta poseti jedan korisnik")
    save(fig, "visits_per_user.png")


def plot_visits_per_vroot(per_vroot: pd.Series) -> None:
    bins = np.logspace(0, np.log10(per_vroot.max()), 30)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_vroot, bins=bins, color="#DD8452")
    ax.set_xscale("log")
    ax.set_xlabel("broj poseta oblasti (log)")
    ax.set_ylabel("broj oblasti")
    ax.set_title("Koliko poseta dobija jedna oblast sajta")
    save(fig, "visits_per_vroot.png")


def plot_zipf(popularity: pd.DataFrame) -> None:
    visited = popularity[popularity["n_visits"] > 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(visited["rank"], visited["n_visits"], marker=".", linestyle="none")
    ax.set_xlabel("rang oblasti (log)")
    ax.set_ylabel("broj poseta (log)")
    ax.set_title("Zipfova raspodela popularnosti oblasti")
    save(fig, "zipf_vroots.png")


def plot_coverage(popularity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(popularity["rank"], popularity["cumulative_share"], color="#55A868")
    for level in (0.5, 0.8, 0.9, 0.95):
        rank = int((popularity["cumulative_share"] >= level).idxmax()) + 1
        ax.axhline(level, color="grey", linewidth=0.6, linestyle=":")
        ax.annotate(f"{level:.0%} pri {rank} oblasti", (rank, level),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("broj najpopularnijih oblasti")
    ax.set_ylabel("udeo svih poseta")
    ax.set_title("Kumulativna pokrivenost poseta")
    save(fig, "vroot_coverage.png")


def plot_jaccard(matrix: pd.DataFrame) -> None:
    # Dijagonala je uvek 1 i sabila bi skalu boja, pa se izostavlja.
    mask = np.eye(len(matrix), dtype=bool)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(matrix, mask=mask, cmap="rocket_r", square=True, linewidths=0.3,
                cbar_kws={"label": "Žakarova sličnost"}, ax=ax)
    ax.set_title(f"Sličnost {len(matrix)} najpopularnijih oblasti")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    save(fig, "vroot_jaccard.png")


def main() -> None:
    ensure_dirs()
    vroots = pd.read_csv(INTERIM_DIR / "vroots.csv")
    users = pd.read_csv(INTERIM_DIR / "users.csv")
    visits = pd.read_csv(INTERIM_DIR / "visits.csv")

    train_users = users.loc[users["source"] == "train", "user"]
    train_visits = visits[visits["user"].isin(train_users)]

    per_user = train_visits.groupby("user").size()
    per_vroot = train_visits.groupby("vroot_id").size()
    popularity = vroot_popularity(vroots, train_visits)

    tables = {
        "eda_basic.csv": basic_stats(vroots, users, visits),
        "eda_distributions.csv": pd.concat(
            [
                distribution_summary(per_user, "visits_per_user"),
                distribution_summary(per_vroot, "visits_per_vroot"),
            ],
            ignore_index=True,
        ),
        "eda_vroot_popularity.csv": popularity,
        "eda_threshold_grid.csv": pd.concat(
            [
                threshold_grid(train_visits, "train", VROOT_OPTIONS, USER_OPTIONS),
                threshold_grid(visits, "all", VROOT_OPTIONS, USER_OPTIONS),
            ],
            ignore_index=True,
        ),
    }

    print("tabele:")
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / name, index=False)
        print(f"  {name} ({len(table)} redova)")

    print("grafici:")
    plot_visits_per_user(per_user)
    plot_visits_per_vroot(per_vroot)
    plot_zipf(popularity)
    plot_coverage(popularity)
    plot_jaccard(vroot_jaccard(vroots, train_visits, HEATMAP_TOP_N))

    print(f"\n{tables['eda_basic.csv'].to_string(index=False)}")


if __name__ == "__main__":
    main()
