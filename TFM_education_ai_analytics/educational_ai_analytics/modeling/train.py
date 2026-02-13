import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from educational_ai_analytics.config import PROCESSED_DATA_DIR, MODELS_DIR
from educational_ai_analytics.modeling.params import AE_PARAMS
from educational_ai_analytics.modeling import TimeSeriesAutoencoder

app = typer.Typer()

def load_data_for_training(path: Path):
    """Carga y prepara los tensores para el entrenamiento."""
    logger.info(f"Cargando datos desde: {path}")
    
    clicks = pd.read_csv(path / "ts_clicks.csv", index_col=0).sort_index()
    perf = pd.read_csv(path / "ts_performance.csv", index_col=0).sort_index()
    proc = pd.read_csv(path / "ts_procrastination.csv", index_col=0).sort_index()
    static = pd.read_csv(path / "static_features.csv", index_col=0).sort_index()
    
    n_students = len(clicks)

    def to_3d_tensor(df, num_vars):
        return df.values.reshape(n_students, num_vars, 40).transpose(0, 2, 1)

    X_temp = np.concatenate([
        to_3d_tensor(clicks, 4),
        to_3d_tensor(perf, 6),
        to_3d_tensor(proc, 1)
    ], axis=2)
    
    X_static = static.values
    return X_temp, X_static

@app.command()
def main(
    epochs: int = AE_PARAMS.epochs,
    batch_size: int = AE_PARAMS.batch_size,
    model_name: str = "best_autoencoder.keras"
):
    """
    Entrena el TimeSeriesAutoencoder usando los datos de data/2_processed/
    """
    # 0. Configuración GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU Activa: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error configuración GPU: {e}")

    # 1. Rutas
    train_path = PROCESSED_DATA_DIR / "training" / "features"
    val_path = PROCESSED_DATA_DIR / "validation" / "features"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Cargar datos
    X_train_temp, X_train_static = load_data_for_training(train_path)
    X_val_temp, X_val_static = load_data_for_training(val_path)

    # 3. Instanciar Modelo
    model = TimeSeriesAutoencoder(
        latent_dim=AE_PARAMS.latent_dim,
        timesteps=AE_PARAMS.timesteps,
        temp_features=AE_PARAMS.temp_features,
        static_features=AE_PARAMS.static_features,
        lstm_units=AE_PARAMS.lstm_units,
        dense_static_units=AE_PARAMS.dense_static_units,
        activation=AE_PARAMS.activation
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AE_PARAMS.learning_rate),
        loss='mse'
    )

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(
            MODELS_DIR / model_name,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=AE_PARAMS.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=AE_PARAMS.reduce_lr_factor,
            patience=AE_PARAMS.reduce_lr_patience,
            min_lr=AE_PARAMS.min_learning_rate,
            verbose=1
        )
    ]

    # 5. Entrenar
    logger.info(f"🚀 Iniciando entrenamiento: {epochs} épocas, batch {batch_size}")
    model.fit(
        x=[X_train_temp, X_train_static],
        y=X_train_temp,
        validation_data=([X_val_temp, X_val_static], X_val_temp),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.success(f"✨ Modelo guardado en {MODELS_DIR / model_name}")

if __name__ == "__main__":
    app()
