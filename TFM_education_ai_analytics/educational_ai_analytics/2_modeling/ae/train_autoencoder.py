import os
import warnings
# Silence Protobuf and TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from educational_ai_analytics.config import (
    FEATURES_DATA_DIR,
    EMBEDDINGS_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    W_WINDOWS,
)

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


def load_features(path: Path, W: int):
    """
    Carga y concatena features para el AE (Static + Dynamic UptoW).
    """
    X0_path = path / "day0_static_features.csv"
    Xdyn_path = path / "ae_uptow_features" / f"ae_uptow_features_w{W:02d}.csv"

    if not X0_path.exists():
        raise FileNotFoundError(f"Falta {X0_path}")
    if not Xdyn_path.exists():
        raise FileNotFoundError(f"Falta {Xdyn_path}")

    df0 = pd.read_csv(X0_path, index_col=0)
    dfdyn = pd.read_csv(Xdyn_path, index_col=0)

    # Concat y alineación
    df = pd.concat([df0, dfdyn.reindex(df0.index)], axis=1).fillna(0.0)
    X = df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
    return X


def make_dataset(X: np.ndarray, batch_size: int, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, X))
    if training:
        ds = ds.shuffle(min(len(X), 100000), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


@app.command()
def main(
    epochs: int = AE_PARAMS.epochs,
    batch_size: int = AE_PARAMS.batch_size,
    seed: int = 42,
    windows_str: str = "",
):
    """
    Entrena un ÚNICO Autoencoder apilando todas las semanas (W_WINDOWS).
    Esto crea un espacio latente común y alineado para todas las fases del curso.
    """
    _configure_gpu()
    _set_seed(seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if windows_str:
        windows = [int(w.strip()) for w in windows_str.split(",") if w.strip()]
    else:
        windows = [int(w) for w in W_WINDOWS]
    
    logger.info(f"🧠 Preparando entrenamiento GLOBAL apilando ventanas: {windows}")

    # 1) Carga y Apilado Vertical
    X_train_list = []
    X_val_list = []

    for W in windows:
        try:
            xt = load_features(FEATURES_DATA_DIR / "training", W=W)
            xv = load_features(FEATURES_DATA_DIR / "validation", W=W)
            X_train_list.append(xt)
            X_val_list.append(xv)
            logger.info(f"   📥 W{W:02d} cargada: train={xt.shape}, val={xv.shape}")
        except Exception as e:
            logger.warning(f"   ⚠️ Error cargando W{W:02d}: {e}")

    if not X_train_list:
        logger.error("❌ No hay datos para entrenar.")
        return

    X_train_full = np.vstack(X_train_list)
    X_val_full = np.vstack(X_val_list)

    logger.info(f"📊 Dataset GLOBAL apilado: Train={X_train_full.shape}, Val={X_val_full.shape}")

    # 2) Definición del Modelo
    model = StudentProfileAutoencoder(input_dim=X_train_full.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AE_PARAMS.learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    model_path = MODELS_DIR / "ae_best_global.keras"
    # Guardar en carpeta oculta para no ensuciar reports/ae
    hist_dir = REPORTS_DIR / "ae" / ".history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    history_path = hist_dir / "ae_training_history_global.csv"

    # 3) Callbacks
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=1),
        EarlyStopping(
            monitor="val_loss",
            patience=AE_PARAMS.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=AE_PARAMS.reduce_lr_factor,
            patience=AE_PARAMS.reduce_lr_patience,
            min_lr=AE_PARAMS.min_learning_rate,
            verbose=1,
        ),
    ]

    # 4) Entrenamiento
    ds_train = make_dataset(X_train_full, batch_size=batch_size, training=True)
    ds_val = make_dataset(X_val_full, batch_size=batch_size, training=False)

    logger.info(f"🚀 Iniciando FIT Global | epochs={epochs} | latent_dim={AE_PARAMS.latent_dim}")
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=1)

    # 5) Guardar Historial
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    logger.success(f"✨ Entrenamiento GLOBAL completado.")
    logger.info(f"💾 Modelo: {model_path}")
    logger.info(f"📈 History: {history_path}")


if __name__ == "__main__":
    app()
