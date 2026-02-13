import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA
from loguru import logger
import typer

from educational_ai_analytics.config import PROCESSED_DATA_DIR, EMBEDDINGS_DATA_DIR, MODELS_DIR
from educational_ai_analytics.modeling.params import AE_PARAMS
from educational_ai_analytics.modeling import TimeSeriesAutoencoder

app = typer.Typer()

def load_processed_tensors(path: Path):
    logger.info(f"Cargando features procesadas desde: {path}")
    
    # Los datos en 2_processed YA vienen normalizados 0-1 por features.py
    clicks = pd.read_csv(path / "ts_clicks.csv", index_col=0).sort_index()
    perf = pd.read_csv(path / "ts_performance.csv", index_col=0).sort_index()
    proc = pd.read_csv(path / "ts_procrastination.csv", index_col=0).sort_index()
    static = pd.read_csv(path / "static_features.csv", index_col=0).sort_index()
    
    n_students = len(clicks)
    student_ids = clicks.index

    def to_3d_tensor(df, num_vars):
        # (N, Vars*40) -> (N, 40, Vars)
        return df.values.reshape(n_students, num_vars, 40).transpose(0, 2, 1)

    tensor_clicks = to_3d_tensor(clicks, 4)
    tensor_perf = to_3d_tensor(perf, 6)
    tensor_proc = to_3d_tensor(proc, 1)

    # Concatenamos bloques (Total: 11 variables temporales)
    X_temp = np.concatenate([tensor_clicks, tensor_perf, tensor_proc], axis=2)
    X_static = static.values
    
    return X_temp, X_static, student_ids

@app.command()
def main(
    model_name: str = "best_autoencoder.keras",
    pca_only: bool = False
):
    """
    Genera los embeddings latentes usando PCA y/o el Autoencoder entrenado.
    """
    # 1. Preparar rutas
    features_path = PROCESSED_DATA_DIR / "training" / "features"
    os.makedirs(EMBEDDINGS_DATA_DIR / "training", exist_ok=True)
    
    # 2. Cargar datos (YA NORMALIZADOS)
    X_temp, X_static, student_ids = load_processed_tensors(features_path)
    N, T, F = X_temp.shape
    
    # 3. GENERAR EMBEDDINGS PCA
    logger.info(f"Generando embeddings PCA ({AE_PARAMS.latent_dim} dimensiones)...")
    X_flat = X_temp.reshape(N, -1)
    X_pca_input = np.concatenate([X_flat, X_static], axis=1)
    
    pca = PCA(n_components=AE_PARAMS.latent_dim, random_state=42)
    latent_pca = pca.fit_transform(X_pca_input)
    
    df_pca = pd.DataFrame(
        latent_pca, 
        index=student_ids, 
        columns=[f"pca_dim_{i}" for i in range(AE_PARAMS.latent_dim)]
    )
    df_pca.to_csv(EMBEDDINGS_DATA_DIR / "training" / "latent_pca.csv")
    
    if pca_only:
        logger.success("Embeddings PCA generados correctamente. Saltando Autoencoder.")
        return

    # 4. GENERAR EMBEDDINGS AUTOENCODER
    model_path = MODELS_DIR / model_name
    # Fallback si no especifican y no existe el basico
    if not model_path.exists() and model_name == "best_autoencoder.keras":
        model_path = MODELS_DIR / "best_autoencoder_final.keras"

    if model_path.exists():
        logger.info(f"Cargando modelo Autoencoder desde: {model_path}")
        model = TimeSeriesAutoencoder(
            latent_dim=AE_PARAMS.latent_dim,
            timesteps=AE_PARAMS.timesteps,
            temp_features=AE_PARAMS.temp_features,
            static_features=AE_PARAMS.static_features,
            lstm_units=AE_PARAMS.lstm_units,
            dense_static_units=AE_PARAMS.dense_static_units,
            activation=AE_PARAMS.activation
        )
        # Build model
        _ = model([X_temp[:1], X_static[:1]])
        model.load_weights(str(model_path))
        
        logger.info("Generando embeddings Autoencoder...")
        latent_ae = model.encode([X_temp, X_static]).numpy()
        
        df_ae = pd.DataFrame(
            latent_ae, 
            index=student_ids, 
            columns=[f"ae_dim_{i}" for i in range(AE_PARAMS.latent_dim)]
        )
        df_ae.to_csv(EMBEDDINGS_DATA_DIR / "training" / "latent_ae.csv")
        logger.success(f"✨ Embeddings guardados en {EMBEDDINGS_DATA_DIR}/training/")
    else:
        logger.error(f"❌ No se encontró el modelo en {model_path}. Solo se guardó PCA.")

if __name__ == "__main__":
    app()
