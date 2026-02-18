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

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"  # Legacy/General
AE_REPORTS_DIR = REPORTS_DIR / "ae"
CLUSTERING_REPORTS_DIR = REPORTS_DIR / "clustering"
TRANSFORMERS_REPORTS_DIR = REPORTS_DIR / "transformers"


OULAD_DATASET_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"

# Feature Engineering Config
W_WINDOWS = [10, 14, 18, 24]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
