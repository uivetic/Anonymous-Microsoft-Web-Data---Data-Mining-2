from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "anonymous+microsoft+web+data"
RAW_TRAIN = RAW_DIR / "anonymous-msweb.data"
RAW_TEST = RAW_DIR / "anonymous-msweb.test"
RAW_INFO = RAW_DIR / "anonymous-msweb.info"

INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

REPORT_DIR = ROOT / "report"

RESULTS_REGISTRY = TABLES_DIR / "results.csv"

RANDOM_STATE = 42

SAMPLE_SIZE = 8000
ROCK_SAMPLE_SIZE = 3000

MIN_VROOT_VISITS = 30

MIN_USER_VISITS = 2


def ensure_dirs() -> None:
    for path in (
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
