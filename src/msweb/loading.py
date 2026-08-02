import csv
from pathlib import Path

import pandas as pd

from msweb.config import RAW_TEST, RAW_TRAIN

SOURCES = {"train": RAW_TRAIN, "test": RAW_TEST}


def load_dst(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vroots: list[tuple[int, str, str]] = []
    user_ids: list[int] = []
    visits: list[tuple[int, int]] = []
    current_user: int | None = None

    with path.open(encoding="latin-1", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            line_type = row[0]
            if line_type == "A":
                vroots.append((int(row[1]), row[3], row[4]))
            elif line_type == "C":
                current_user = int(row[2])
                user_ids.append(current_user)
            elif line_type == "V":
                if current_user is None:
                    raise ValueError(
                        f"{path.name}, linija {line_number}: "
                        f"poseta pre prve 'C' linije"
                    )
                visits.append((current_user, int(row[1])))

    return (
        pd.DataFrame(vroots, columns=["vroot_id", "title", "url"]),
        pd.DataFrame({"user_id": user_ids}),
        pd.DataFrame(visits, columns=["user_id", "vroot_id"]),
    )


def load_both() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vroots_by_source: dict[str, pd.DataFrame] = {}
    user_frames: list[pd.DataFrame] = []
    visit_frames: list[pd.DataFrame] = []

    for source, path in SOURCES.items():
        vroots, users, visits = load_dst(path)
        # Oba fajla navode istih 294 vroot-a, ali razlicitim redosledom.
        vroots = vroots.sort_values("vroot_id").reset_index(drop=True)
        vroots_by_source[source] = vroots

        # ID-jevi 10001-15000 postoje u oba fajla, ali oznacavaju razlicite
        # korisnike, pa im je potreban kljuc koji nosi i izvor.
        users = users.assign(
            source=source, user=source + "_" + users["user_id"].astype(str)
        )
        visits = visits.assign(user=source + "_" + visits["user_id"].astype(str))

        user_frames.append(users[["user", "user_id", "source"]])
        visit_frames.append(visits[["user", "vroot_id"]])

    train_vroots, test_vroots = vroots_by_source["train"], vroots_by_source["test"]
    if not train_vroots.equals(test_vroots):
        raise ValueError("definicije vroot-ova se razlikuju izmedju .data i .test")

    all_users = pd.concat(user_frames, ignore_index=True)
    all_visits = pd.concat(visit_frames, ignore_index=True)
    _validate(train_vroots, all_users, all_visits)
    return train_vroots, all_users, all_visits


def _validate(
    vroots: pd.DataFrame, users: pd.DataFrame, visits: pd.DataFrame
) -> None:
    if vroots["vroot_id"].duplicated().any():
        raise ValueError("duplirani vroot_id")
    if users["user"].duplicated().any():
        raise ValueError("duplirani kljuc korisnika")
    if visits.duplicated().any():
        raise ValueError("duplirani par (user, vroot_id)")

    unknown_vroots = set(visits["vroot_id"]) - set(vroots["vroot_id"])
    if unknown_vroots:
        raise ValueError(f"posete ka nepoznatim vroot-ovima: {sorted(unknown_vroots)}")

    unknown_users = set(visits["user"]) - set(users["user"])
    if unknown_users:
        raise ValueError(f"posete nepoznatih korisnika: {len(unknown_users)}")
