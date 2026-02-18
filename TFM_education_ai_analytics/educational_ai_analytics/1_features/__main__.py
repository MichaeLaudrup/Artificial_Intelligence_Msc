import os
import warnings
# Silence Protobuf and TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import logging
import shutil
import pandas as pd
from pathlib import Path
import sys

# Añadimos el directorio actual al path para poder importar módulos locales con nombres numéricos
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from day0_static_features import DayZeroFeaturesBuilder

# ✅ NUEVO: builder dinámico acumulativo uptoW (tipo agg_week) con normalización curso-convocatoria
# Guarda: data/3_features/<split>/ae_uptow_features/ae_uptow_features_wXX.csv
from ae_uptow_features import AEUptoWFeaturesBuilder
from check_features_integrity import check_ae_features_integrity

# Importamos las constantes de config
# Para que funcione el import de config fuera de la carpeta, aseguramos que la raíz esté en el path
sys.path.append(str(current_dir.parents[1]))
from educational_ai_analytics.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR, W_WINDOWS

TARGET_MAP = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}

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


def _ensure_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    if "unique_id" in df.columns:
        return df
    needed = {"id_student", "code_module", "code_presentation"}
    if not needed.issubset(df.columns):
        return df  # robusto
    out = df.copy()
    out["unique_id"] = (
        out["id_student"].astype(str)
        + "_"
        + out["code_module"].astype(str)
        + "_"
        + out["code_presentation"].astype(str)
    )
    return out


def main():
    logger.info("🎬 Iniciando Pipeline de Features por SPLITS")

    if FEATURES_DATA_DIR.exists():
        logger.info(f"🧹 Limpiando directorio: {FEATURES_DATA_DIR}")
        shutil.rmtree(FEATURES_DATA_DIR)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Builders
    day0_builder = DayZeroFeaturesBuilder()

    # ✅ Dinámicas AE uptoW (acumulativas por W, normalizadas curso-convocatoria)
    ae_uptow_builder = AEUptoWFeaturesBuilder(
        features_root_dir=FEATURES_DATA_DIR,
        normalize_mode="zscore",  # si quieres ratio usa "ratio_to_mean"
        inter_day_col="date",
        inter_clicks_col="sum_click",
        inter_activity_col="activity_type",
        asm_deadline_col="date",
        asm_submitted_col="date_submitted",
        asm_score_col="score",
    )
    ae_uptow_stats_file = FEATURES_DATA_DIR / "training" / "ae_uptow_features" / "course_stats_ae_uptow.json"

    for split in ["training", "validation", "test"]:
        logger.info(f"\n🚀 Procesando split: {split.upper()}")
        split_path = PROCESSED_DATA_DIR / split
        out_dir = FEATURES_DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            dfs = _load_split_tables(split_path)

            # Asegura unique_id
            for k in ["students", "assessments", "interactions"]:
                dfs[k] = _ensure_unique_id(dfs[k])

            # Base index
            students_idx = dfs["students"].drop_duplicates("unique_id").set_index("unique_id")

            # 1) Day 0 Static Features (Transformer)
            X_day0 = day0_builder.process_pipeline(
                dfs["students"],
                dfs["interactions"],
                fit=(split == "training"),
            )
            X_day0 = X_day0.reindex(students_idx.index).fillna(0.0)
            X_day0.to_csv(out_dir / "day0_static_features.csv")
            logger.info(f"   ✅ Day 0 Static Features: {X_day0.shape}")

            # 2) Target
            target = students_idx[["final_result"]].copy()
            target["final_result"] = target["final_result"].map(TARGET_MAP).fillna(0).astype(int)
            target = target.reindex(X_day0.index).fillna(0).astype(int)
            target.to_csv(out_dir / "target.csv")
            logger.info("   🎯 Target guardado.")

            # 3) ✅ AE UptoW Features (acumulativas por W, normalizadas curso-convocatoria)
            logger.info(f"   ⏳ Generando AE UptoW Features para W={W_WINDOWS}...")
            if split != "training" and ae_uptow_stats_file.exists():
                ae_uptow_builder.load_stats(ae_uptow_stats_file)

            saved_uptow = ae_uptow_builder.build_for_split(
                split=split,
                df_students=dfs["students"],
                df_interactions=dfs["interactions"],
                df_assessments=dfs["assessments"],
                windows=W_WINDOWS,
                fit=(split == "training"),
                save_stats=True,
                min_weeks=1,
            )

            # opcional: mostrar shape sin guardar copia redundante en la raíz
            if saved_uptow:
                maxW = max(saved_uptow.keys())
                # Leemos solo el header para saber el número de columnas
                sample_dyn = pd.read_csv(saved_uptow[maxW], index_col=0, nrows=0)
                logger.info(f"   ✅ AE UptoW Dynamic Features (W={maxW}): {sample_dyn.shape[1]} features")
                
                # 5) ✅ Control de Integridad (sobre maxW)
                check_ae_features_integrity(
                    split=split,
                    W=maxW,
                    feats_path=saved_uptow[maxW],
                    students=dfs["students"],
                    interactions=dfs["interactions"],
                    assessments=dfs["assessments"]
                )

        except Exception as e:
            logger.error(f"❌ Error en split {split}: {e}", exc_info=True)
            continue

    logger.info("\n✅ Pipeline completado.")


if __name__ == "__main__":
    main()
