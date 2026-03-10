import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
#logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "0_raw"
INTERIM_DATA_DIR = DATA_DIR / "1_interim"
PROCESSED_DATA_DIR = DATA_DIR / "2_processed"
FEATURES_DATA_DIR = DATA_DIR / "3_features"
EMBEDDINGS_DATA_DIR = DATA_DIR / "4_embeddings"
SEGMENTED_DATA_DIR = DATA_DIR / "5_students_segmented"

MODELS_DIR = PROJ_ROOT / "models"
AE_MODELS_DIR = MODELS_DIR / "ae_models"
CLUSTERING_MODELS_DIR = MODELS_DIR / "clustering_models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"  # Legacy/General
AE_REPORTS_DIR = REPORTS_DIR / "ae"
CLUSTERING_REPORTS_DIR = REPORTS_DIR / "clustering"
TRANSFORMERS_REPORTS_DIR = REPORTS_DIR / "transformers"


OULAD_DATASET_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"

# Transformer profile used by the project by default.
# Valid values:
# - "binary": 2 classes | success_vs_risk
# - "binary_paper": 2 classes | paper baseline
# - "trinary": 3 classes
# - "quaternary": 4 classes
# Environment variable TFM_TRANSFORMER_PROFILE can still override this value.
TRANSFORMER_PROFILE = os.getenv("TFM_TRANSFORMER_PROFILE", "binary_paper").strip().lower()

# Dataset augmentation toggle.
# When enabled, the training split is expanded by duplicating Withdrawn students
# until that class matches the largest training class. Validation and test stay untouched.
WITH_SYNTHETIC = os.getenv("TFM_WITH_SYNTHETIC", "0").strip().lower() in {"1", "true", "yes", "on"}

# Feature Engineering Config
W_WINDOWS = [1,3,5,8,10,12,15,18,20,24,28]

# Clustering Config (single source of truth)
N_CLUSTERS = 5

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
