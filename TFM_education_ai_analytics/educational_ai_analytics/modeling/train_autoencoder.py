import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from educational_ai_analytics.config import FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR
from educational_ai_analytics.modeling.params import AE_PARAMS
from educational_ai_analytics.modeling import StudentProfileAutoencoder

app = typer.Typer()

def load_static_data(path: Path):
    """Carga las features estáticas para el entrenamiento."""
    logger.info(f"Cargando datos desde: {path}")
    static = pd.read_csv(path / "static_features.csv", index_col=0)
    return static.values

@app.command()
def main(
    epochs: int = AE_PARAMS.epochs,
    batch_size: int = AE_PARAMS.batch_size,
    model_name: str = "best_static_autoencoder.keras"
):
    """
    Entrena el StudentProfileAutoencoder.
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
    train_path = FEATURES_DATA_DIR / "training"
    val_path = FEATURES_DATA_DIR / "validation"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2. Cargar datos
    if not (train_path / "static_features.csv").exists():
        logger.error(f"No se encuentran datos en {train_path}. Ejecuta 'make features' primero.")
        return

    X_train = load_static_data(train_path)
    X_val = load_static_data(val_path)
    
    # Ajustar input_dim dinámicamente por si ha cambiado
    actual_input_dim = X_train.shape[1]
    logger.info(f"Dimensiones de entrada detectadas: {actual_input_dim}")

    # 3. Instanciar Modelo
    model = StudentProfileAutoencoder(
        input_dim=actual_input_dim,
        latent_dim=AE_PARAMS.latent_dim,
        hidden_dims=AE_PARAMS.hidden_dims,
        dropout_rate=AE_PARAMS.dropout_rate
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
            mode='min',
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
    logger.info(f"🚀 Entrenando Autoencoder Estático: {epochs} épocas...")
    history = model.fit(
        x=X_train,
        y=X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Guardar historial para gráficas
    os.makedirs(REPORTS_DIR, exist_ok=True)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(REPORTS_DIR / "ae_training_history.csv", index=False)
    logger.info(f"📊 Historial de entrenamiento guardado en {REPORTS_DIR / 'ae_training_history.csv'}")

    logger.success(f"✨ Modelo guardado en {MODELS_DIR / model_name}")

if __name__ == "__main__":
    app()
