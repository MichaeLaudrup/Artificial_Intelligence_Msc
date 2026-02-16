import logging
import shutil
import pandas as pd
from pathlib import Path

from educational_ai_analytics.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR
from educational_ai_analytics.features import FeatureEngineer, build_day0_static_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def _load_split_tables(split_path: Path) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for name in ["students", "assessments", "interactions"]:
        fp = split_path / f"{name}.csv"
        if not fp.exists():
            raise FileNotFoundError(str(fp))
        dfs[name] = pd.read_csv(fp)
    return dfs

def main():
    """
    Pipeline automatizado de Feature Engineering.
    Genera en FEATURES_DATA_DIR:
      - engineered_features.csv (Clustering Features)
      - target.csv
      - day0_static_features.csv (NUEVO: Static Features para Transformers/Hybrid)
    """
    logger.info("🎬 Iniciando Pipeline de Features (Automatizado)")

    if FEATURES_DATA_DIR.exists():
        logger.info(f"🧹 Limpiando directorio: {FEATURES_DATA_DIR}")
        shutil.rmtree(FEATURES_DATA_DIR)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    engineer = FeatureEngineer()

    for split in ["training", "validation", "test"]:
        logger.info(f"\n🚀 Procesando split: {split.upper()}")
        
        split_path = PROCESSED_DATA_DIR / split
        out_dir = FEATURES_DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            dfs = _load_split_tables(split_path)
        except FileNotFoundError:
            logger.error(f"❌ Faltan archivos en {split_path}. Ejecuta 'make data' primero.")
            return

        # Asegura unique_id
        for k in ["students", "assessments", "interactions"]:
            dfs[k] = engineer._prepare_unique_id(dfs[k])

        # 1. Features de Clustering (Originales)
        _, X_eng = engineer.process_split(
            dfs["students"], dfs["assessments"], dfs["interactions"], fit=(split == "training")
        )
        X_eng.to_csv(out_dir / "engineered_features.csv")
        logger.info(f"   ✅ Clustering Features: {X_eng.shape}")

        # 2. Features Estáticas Day 0 (NUEVAS)
        # Alineamos con X_eng para consistencia
        X_day0 = build_day0_static_features(
            dfs["students"], 
            dfs["interactions"],
            index_uids=X_eng.index
        )
        X_day0.to_csv(out_dir / "day0_static_features.csv")
        logger.info(f"   ✅ Day 0 Static Features: {X_day0.shape}")

        # 3. Target
        students_idx = dfs["students"].set_index("unique_id")
        target = students_idx.loc[X_eng.index, ["final_result"]].copy()
        target["final_result"] = target["final_result"].map(engineer.target_map).fillna(0).astype(int)
        target.to_csv(out_dir / "target.csv")
        
        logger.info(f"   🎯 Target guardado.")

    logger.info("\n✅ Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
