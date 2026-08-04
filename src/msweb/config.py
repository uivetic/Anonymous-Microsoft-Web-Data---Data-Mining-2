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

TSNE_PERPLEXITY = 30
# Za O(n^2) algoritme (hijerarhijsko, ROCK) po potrebi kasnije.
ROCK_SAMPLE_SIZE = 3000

MODEL_SOURCE = "train"
MIN_USER_VISITS = 2

FEATURE_SETS_DIR = PROCESSED_DIR / "feature_sets"

# Pragovi skupova atributa
A1_MIN_VROOT_USERS = 10
A2_TOP_N = 50
A3_MIN_VARIANCE = 0.01
A4_MAX_PHI = 0.9
A5_VARIANCE_RATIO = 0.90
A5_MAX_COMPONENTS = 150

FEATURE_SET_NAMES = ("A0", "A1", "A2", "A3", "A4", "A5")

KMEANS_K_MIN = 2
KMEANS_K_MAX = 60
KMEANS_SELECTED_K = 25
KMEANS_N_INIT = 10


def ensure_dirs() -> None:
    for path in (
        INTERIM_DIR,
        PROCESSED_DIR,
        FEATURE_SETS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
