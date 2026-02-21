import logging
import shutil
from pathlib import Path

import pandas as pd

from educational_ai_analytics.config import W_WINDOWS

from transformer_features_builder import TransformerFeaturesBuilder


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


# Rutas (siguiendo tu convención)
PROCESSED_DATA_DIR = Path("/workspace/TFM_education_ai_analytics/data/2_processed")
FEATURES_DATA_DIR = Path("/workspace/TFM_education_ai_analytics/data/3_features")
SEGMENTED_DATA_DIR = Path("/workspace/TFM_education_ai_analytics/data/5_students_segmented")
TRANSFORMER_OUT_DIR = Path("/workspace/TFM_education_ai_analytics/data/6_transformer_features")


def _load_split_tables(split_path: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in ["students", "interactions"]:
        fp = split_path / f"{name}.csv"
        if not fp.exists():
            raise FileNotFoundError(str(fp))
        dfs[name] = pd.read_csv(fp)
    return dfs


def main():
    logger.info("🎬 Iniciando generación de Transformer Features (6_transformer_features)")

    # Limpieza dir salida
    if TRANSFORMER_OUT_DIR.exists():
        logger.info(f"🧹 Limpiando directorio: {TRANSFORMER_OUT_DIR}")
        shutil.rmtree(TRANSFORMER_OUT_DIR)
    TRANSFORMER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    builder = TransformerFeaturesBuilder(
        out_root_dir=TRANSFORMER_OUT_DIR,
        segmented_root_dir=SEGMENTED_DATA_DIR,
        features_root_dir=FEATURES_DATA_DIR,
        windows=W_WINDOWS,
    )

    for split in ["training", "validation", "test"]:
        logger.info(f"\n🚀 Split: {split.upper()}")
        split_path = PROCESSED_DATA_DIR / split

        dfs = _load_split_tables(split_path)

        # fit sólo en training (para fijar activities_global y guardarla en meta)
        fit = (split == "training")

        saved = builder.build_for_split(
            split=split,
            students_df=dfs["students"],
            interactions_df=dfs["interactions"],
            fit=fit,
        )

        if saved:
            logger.info(f"✅ Guardados {len(saved)} archivos .npz en {TRANSFORMER_OUT_DIR / split}")

    logger.info("\n✅ Listo. Transformer features generadas.")


if __name__ == "__main__":
    main()