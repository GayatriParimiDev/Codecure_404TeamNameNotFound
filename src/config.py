from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "toxicity_pipeline.joblib"

TOX21_PATH = DATA_DIR / "tox21.csv"
ZINC_PATH = DATA_DIR / "250k_rndm_zinc_drugs_clean_3.csv"

TARGETS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

FP_BITS = 1024
KMEANS_CLUSTERS = 20
NN_NEIGHBORS = 5
RANDOM_STATE = 42
MIN_PRECISION = 0.15
USE_SMOTE = False
SMOTE_K_NEIGHBORS = 5
ZINC_SUPPORT_SAMPLE_SIZE = 25000
DEFAULT_SCREEN_SAMPLE_SIZE = 2000
