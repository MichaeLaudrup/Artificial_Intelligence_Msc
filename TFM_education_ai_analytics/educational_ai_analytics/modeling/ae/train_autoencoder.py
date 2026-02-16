import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from educational_ai_analytics.config import FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR
from .hyperparams import AE_PARAMS
from .autoencoder import StudentProfileAutoencoder

app = typer.Typer()


def _set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU Activa: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error configuración GPU: {e}")


def load_features(path: Path, prefer: str = "engineered_features.csv"):
    """Carga features para el AE."""
    prefer_path = path / prefer
    fallback_path = path / "static_features.csv"

    if prefer_path.exists():
        df = pd.read_csv(prefer_path, index_col=0)
        used = prefer
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path, index_col=0)
        used = "static_features.csv"
    else:
        raise FileNotFoundError(f"No se encontraron features en {path}")

    X = df.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    return X, df.index, used


def make_dataset(X: np.ndarray, batch_size: int, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, X))
    if training:
        ds = ds.shuffle(min(len(X), 50000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def save_embeddings(model: StudentProfileAutoencoder, X: np.ndarray, index: pd.Index, out_path: Path, batch_size: int = 1024):
    Z = model.get_embeddings(X, batch_size=batch_size).numpy()
    z_cols = [f"z_{i:02d}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, index=index, columns=z_cols).to_csv(out_path)
    logger.info(f"🧬 Embeddings guardados: {out_path} | shape={Z.shape}")


@app.command()
def main(
    epochs: int = AE_PARAMS.epochs,
    batch_size: int = AE_PARAMS.batch_size,
    model_name: str = "best_static_autoencoder.keras",
    seed: int = 42,
):
    """Entrena el AE usando los parámetros centralizados en params.py."""
    _set_seed(seed)
    _configure_gpu()

    # Cargar datos
    X_train, idx_train, _ = load_features(FEATURES_DATA_DIR / "training")
    X_val, idx_val, _ = load_features(FEATURES_DATA_DIR / "validation")

    # Instanciar Modelo (usa AE_PARAMS por defecto en su __init__)
    model = StudentProfileAutoencoder(input_dim=X_train.shape[1])

    # Compilación (usando parámetros de params.py si existieran allí, o defaults seguros)
    loss_fn = tf.keras.losses.Huber(delta=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AE_PARAMS.learning_rate),
        loss=loss_fn,
    )

    # Callbacks simplificados
    callbacks = [
        ModelCheckpoint(MODELS_DIR / model_name, save_best_only=True, monitor="val_loss", verbose=1),
        EarlyStopping(monitor="val_loss", patience=AE_PARAMS.early_stopping_patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=AE_PARAMS.reduce_lr_factor, patience=AE_PARAMS.reduce_lr_patience, verbose=1),
    ]

    # Entrenar
    ds_train = make_dataset(X_train, batch_size=batch_size, training=True)
    ds_val = make_dataset(X_val, batch_size=batch_size, training=False)
    
    logger.info(f"🚀 Entrenando Autoencoder: epochs={epochs} | latent_dim={AE_PARAMS.latent_dim}")
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=1)

    # Reportes
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame(history.history).to_csv(REPORTS_DIR / "ae_training_history.csv", index=False)

    logger.success(f"✨ Proceso completado. Modelo en {MODELS_DIR / model_name}")


if __name__ == "__main__":
    app()