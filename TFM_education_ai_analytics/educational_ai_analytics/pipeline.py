"""
Pipeline Orquestador del Proyecto Educational AI Analytics.

Ejecuta los pasos del pipeline en orden:
  1. Dataset:  Descarga, Merge, Limpieza, Split (train/val/test)
  2. Features: Ingeniería de Características (Fit on Train, Transform All)
  3. Train:    Entrenamiento del Modelo (Futuro)

Uso:
  python -m educational_ai_analytics.pipeline          # Pipeline completo
  python -m educational_ai_analytics.pipeline --step dataset   # Solo dataset
  python -m educational_ai_analytics.pipeline --step features  # Solo features
"""

import typer
from pathlib import Path
from loguru import logger

from educational_ai_analytics.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def step_dataset():
    """Paso 1: Descarga, Merge, Limpieza Estructural, Split, Imputación."""
    from educational_ai_analytics.dataset import main as dataset_main
    logger.info("=" * 60)
    logger.info("PASO 1: DATASET (Descarga + Procesamiento + Split)")
    logger.info("=" * 60)
    dataset_main(
        input_path=RAW_DATA_DIR / "OULAD_dataset" / "oulad_dataset.zip",
        output_path=PROCESSED_DATA_DIR / "dataset.csv",
    )


def step_features():
    """Paso 2: Ingeniería de Características."""
    from educational_ai_analytics.features import run_feature_pipeline
    logger.info("=" * 60)
    logger.info("PASO 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    run_feature_pipeline()


def step_train():
    """Paso 3: Entrenamiento del Modelo (Placeholder)."""
    logger.info("=" * 60)
    logger.info("PASO 3: ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 60)
    logger.warning("Todavía no implementado. Próximamente...")
    # from educational_ai_analytics.modeling.train import run_training
    # run_training()


STEPS = {
    "dataset": step_dataset,
    "features": step_features,
    "train": step_train,
}


@app.command()
def main(
    step: str = typer.Option(
        None, 
        help="Paso específico a ejecutar: 'dataset', 'features', 'train'. Si no se indica, ejecuta todo."
    ),
    skip_dataset: bool = typer.Option(
        False,
        "--skip-dataset",
        help="Salta el paso de dataset (útil si ya tienes los datos descargados y procesados)."
    ),
):
    """
    Orquesta el pipeline completo de Educational AI Analytics.
    """
    logger.info("🚀 Iniciando Pipeline de Educational AI Analytics")
    
    if step:
        # Ejecutar solo un paso específico
        if step not in STEPS:
            logger.error(f"Paso '{step}' no reconocido. Opciones: {list(STEPS.keys())}")
            raise typer.Exit(1)
        STEPS[step]()
    else:
        # Pipeline completo
        if not skip_dataset:
            step_dataset()
        else:
            logger.info("⏭️  Saltando paso de Dataset (--skip-dataset)")
        
        step_features()
        step_train()
    
    logger.success("✅ Pipeline finalizado correctamente.")


if __name__ == "__main__":
    app()
