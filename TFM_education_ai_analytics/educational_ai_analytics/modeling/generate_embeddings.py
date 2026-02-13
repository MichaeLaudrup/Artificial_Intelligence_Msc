import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA
from loguru import logger
import typer

from educational_ai_analytics.config import FEATURES_DATA_DIR, EMBEDDINGS_DATA_DIR, MODELS_DIR
from educational_ai_analytics.modeling.params import AE_PARAMS
from educational_ai_analytics.modeling import StudentProfileAutoencoder

app = typer.Typer()

def load_static_features_with_ids(path: Path):
    logger.info(f"Cargando features desde: {path}")
    df = pd.read_csv(path / "static_features.csv", index_col=0)
    return df.values, df.index

@app.command()
def main(
    model_name: str = "best_static_autoencoder.keras",
    pca_only: bool = False
):
    """
    Genera los embeddings latentes (resúmenes) de los estudiantes usando PCA y Autoencoder.
    """
    # 1. Preparar rutas
    features_path = FEATURES_DATA_DIR / "training"
    os.makedirs(EMBEDDINGS_DATA_DIR / "training", exist_ok=True)
    
    # 2. Cargar datos
    if not (features_path / "static_features.csv").exists():
        logger.error(f"No se encuentran features en {features_path}")
        return

    X, student_ids = load_static_features_with_ids(features_path)
    
    # 3. PCA
    logger.info(f"Generando embeddings PCA ({AE_PARAMS.latent_dim}D)...")
    pca = PCA(n_components=AE_PARAMS.latent_dim, random_state=42)
    latent_pca = pca.fit_transform(X)
    
    pd.DataFrame(latent_pca, index=student_ids, 
                columns=[f"pca_{i}" for i in range(AE_PARAMS.latent_dim)]).to_csv(
                    EMBEDDINGS_DATA_DIR / "training" / "latent_pca.csv"
                )
    
    if pca_only:
        logger.success("Embeddings PCA listos.")
        return

    # 4. Autoencoder
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        # Intentar con nombre por defecto anterior si falla
        model_path = MODELS_DIR / "best_autoencoder.keras"

    if model_path.exists():
        logger.info(f"Cargando {model_path}...")
        model = StudentProfileAutoencoder(
            input_dim=X.shape[1],
            latent_dim=AE_PARAMS.latent_dim,
            hidden_dims=AE_PARAMS.hidden_dims,
            dropout_rate=AE_PARAMS.dropout_rate
        )
        # Construir el modelo pasando un batch de prueba
        _ = model(X[:1])
        model.load_weights(str(model_path))
        
        logger.info("Generando representaciones latentes con AE...")
        latent_ae = model.encode(X).numpy()
        pd.DataFrame(latent_ae, index=student_ids, 
                    columns=[f"ae_{i}" for i in range(AE_PARAMS.latent_dim)]).to_csv(
                        EMBEDDINGS_DATA_DIR / "training" / "latent_ae.csv"
                    )
        logger.success(f"✨ Embeddings guardados en {EMBEDDINGS_DATA_DIR}")
    else:
        logger.warning(f"No se encontró el modelo {model_name} en {MODELS_DIR}. Ejecuta 'train_autoencoder' primero.")

if __name__ == "__main__":
    app()
