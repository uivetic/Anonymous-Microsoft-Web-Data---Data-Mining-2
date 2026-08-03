import _bootstrap  # noqa: F401

import sys

try:
    from msweb.config import INTERIM_DIR, ensure_dirs
    from msweb.loading import load_both
except ModuleNotFoundError:
    sys.exit(
        "Nedostaje zavisnost ili paket msweb.\n"
        "Iz korena projekta pokreni:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "zatim: python scripts/01_parse.py"
    )


def main() -> None:
    ensure_dirs()
    vroots, users, visits = load_both()

    vroots.to_csv(INTERIM_DIR / "vroots.csv", index=False)
    users.to_csv(INTERIM_DIR / "users.csv", index=False)
    visits.to_csv(INTERIM_DIR / "visits.csv", index=False)

    per_source = (
        visits.merge(users[["user", "source"]], on="user")
        .groupby("source")
        .agg(visits=("vroot_id", "size"), users=("user", "nunique"))
    )

    print(f"vroots: {len(vroots)}")
    for source, row in per_source.iterrows():
        print(f"{source}: users={row['users']} visits={row['visits']}")
    print(f"ukupno: users={len(users)} visits={len(visits)}")
    print(f"zapisano u {INTERIM_DIR}")


if __name__ == "__main__":
    main()
