from typing import cast

import numpy as np
import pandas as pd

QUANTILES = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def split_by_source(users: pd.DataFrame, visits: pd.DataFrame) -> dict:
    labelled = visits.merge(users[["user", "source"]], on="user")
    parts = {s: g for s, g in labelled.groupby("source")}
    parts["all"] = labelled
    return parts


def basic_stats(
    vroots: pd.DataFrame, users: pd.DataFrame, visits: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for label, subset in split_by_source(users, visits).items():
        per_user = subset.groupby("user").size()
        per_vroot = subset.groupby("vroot_id").size()
        rows.append(
            {
                "dataset": label,
                "n_users": len(per_user),
                "n_vroots": len(vroots),
                "n_vroots_visited": len(per_vroot),
                "n_visits": len(subset),
                "density": len(subset) / (len(per_user) * len(vroots)),
                "visits_per_user_mean": per_user.mean(),
                "visits_per_user_median": per_user.median(),
                "visits_per_user_max": per_user.max(),
                "users_with_one_visit": int((per_user == 1).sum()),
                "share_users_with_one_visit": (per_user == 1).mean(),
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(counts: pd.Series, label: str) -> pd.DataFrame:
    quantiles = counts.quantile(QUANTILES)
    row = {"distribution": label, "n": len(counts), "min": counts.min()}
    row.update({f"q{int(q * 100)}": quantiles[q] for q in QUANTILES})
    row["max"] = counts.max()
    row["mean"] = counts.mean()
    return pd.DataFrame([row])


def vroot_popularity(vroots: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    counts = visits.groupby("vroot_id").size().rename("n_visits")
    table = (
        vroots.merge(counts, on="vroot_id", how="left")
        .fillna({"n_visits": 0})
        .astype({"n_visits": int})
        .sort_values("n_visits", ascending=False)
        .reset_index(drop=True)
    )
    table["rank"] = table.index + 1
    table["share"] = table["n_visits"] / table["n_visits"].sum()
    table["cumulative_share"] = table["share"].cumsum()
    return table


def apply_filters(
    visits: pd.DataFrame, min_vroot_visits: int, min_user_visits: int
) -> pd.DataFrame:
    # Redosled je bitan: uklanjanje vroot-ova smanjuje broj poseta korisnika,
    # pa se prag za korisnike primenjuje tek na preostale posete.
    per_vroot = visits.groupby("vroot_id").size()
    keep = per_vroot[per_vroot >= min_vroot_visits].index
    kept = cast(pd.DataFrame, visits.loc[visits["vroot_id"].isin(keep)].copy())

    per_user = kept.groupby("user").size()
    keep_users = per_user[per_user >= min_user_visits].index
    return cast(pd.DataFrame, kept.loc[kept["user"].isin(keep_users)].copy())


def threshold_grid(
    visits: pd.DataFrame,
    label: str,
    vroot_options: list[int],
    user_options: list[int],
) -> pd.DataFrame:
    n_users_before = visits["user"].nunique()
    n_vroots_before = visits["vroot_id"].nunique()

    rows = []
    for min_vroot in vroot_options:
        for min_user in user_options:
            kept = apply_filters(visits, min_vroot, min_user)
            n_users = kept["user"].nunique()
            n_vroots = kept["vroot_id"].nunique()
            rows.append(
                {
                    "dataset": label,
                    "min_vroot_visits": min_vroot,
                    "min_user_visits": min_user,
                    "n_users": n_users,
                    "n_vroots": n_vroots,
                    "n_visits": len(kept),
                    "kept_users": n_users / n_users_before,
                    "kept_vroots": n_vroots / n_vroots_before,
                    "kept_visits": len(kept) / len(visits),
                    "density": len(kept) / (n_users * n_vroots) if n_users else 0.0,
                }
            )
    return pd.DataFrame(rows)


def vroot_jaccard(
    vroots: pd.DataFrame, visits: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    counts = visits.groupby("vroot_id").size().sort_values(ascending=False)
    top = counts.head(top_n).index
    subset = visits[visits["vroot_id"].isin(top)]

    pairs = subset.merge(subset, on="user")
    intersection = (
        pairs.groupby(["vroot_id_x", "vroot_id_y"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top, columns=top)
    )

    sizes = counts[top].to_numpy()
    union = sizes[:, None] + sizes[None, :] - intersection.to_numpy()
    jaccard = np.zeros_like(union, dtype=float)
    np.divide(intersection.to_numpy(), union, out=jaccard, where=union > 0)

    labels = vroots.set_index("vroot_id").loc[top, "url"]
    return pd.DataFrame(jaccard, index=labels, columns=labels)
