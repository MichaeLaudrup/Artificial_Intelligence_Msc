import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA
import logging
import typer

from educational_ai_analytics.config import FEATURES_DATA_DIR, EMBEDDINGS_DATA_DIR, MODELS_DIR
from educational_ai_analytics.modeling.params import AE_PARAMS
from educational_ai_analytics.modeling.autoencoder import StudentProfileAutoencoder

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def load_features(split_path: Path):
    """Carga features. Intenta engineered_features.csv, fallback a static_features.csv."""
    p = split_path / "engineered_features.csv"
    if not p.exists():
        p = split_path / "static_features.csv"

    if not p.exists():
        raise FileNotFoundError(f"No se encontraron features en {split_path} (engineered/static).")

    logger.info(f"Cargando features desde: {p}")
    df = pd.read_csv(p, index_col=0).replace([np.inf, -np.inf], 0).fillna(0)
    return df.values.astype(np.float32), df.index, p.name


def save_latents(latent: np.ndarray, index: pd.Index, out_csv: Path, prefix: str):
    cols = [f"{prefix}_{i:02d}" for i in range(latent.shape[1])]
    pd.DataFrame(latent, index=index, columns=cols).to_csv(out_csv)
    logger.info(f"Guardado: {out_csv} | shape={latent.shape}")


def encode_in_batches(model: StudentProfileAutoencoder, X: np.ndarray, batch_size: int = 2048):
    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    zs = []
    for xb in ds:
        z = model.encode(xb, training=False)
        zs.append(z)
    return tf.concat(zs, axis=0).numpy()


def load_autoencoder(model_path: Path, input_dim: int):
    """
    Carga AE.
    - Si es .keras, intenta load_model (modelo completo).
    - Si falla, instancia y carga pesos (load_weights).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    if model_path.suffix == ".keras":
        try:
            logger.info("Intentando cargar modelo completo (.keras) ...")
            return tf.keras.models.load_model(
                model_path,
                custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
                compile=False,
            )
        except Exception as e:
            logger.warning(f"Fallo load_model(.keras): {e}. Intento load_weights...")

    # Fallback: pesos
    model = StudentProfileAutoencoder(input_dim=input_dim)
    _ = model(tf.zeros((1, input_dim), dtype=tf.float32), training=False)
    model.load_weights(str(model_path))
    return model


@app.command()
def main(
    model_name: str = "best_static_autoencoder.keras",
    pca_only: bool = False,
    batch_size: int = 2048,
):
    """
    Genera embeddings latentes usando:
    - PCA (fit en train; transform en val/test)
    - Autoencoder (encode en train/val/test)
    """
    os.makedirs(EMBEDDINGS_DATA_DIR, exist_ok=True)

    splits = ["training", "validation", "test"]

    # 1) Cargar features de cada split
    X = {}
    idx = {}
    used = {}
    for split in splits:
        split_path = FEATURES_DATA_DIR / split
        X[split], idx[split], used[split] = load_features(split_path)
        logger.info(f"{split.upper()}: {used[split]} | X={X[split].shape}")

    latent_dim = AE_PARAMS.latent_dim

    # 2) PCA: fit en train y transform en val/test
    logger.info(f"Generando embeddings PCA ({latent_dim}D) [fit train] ...")
    pca = PCA(n_components=latent_dim, random_state=42)
    Z_pca_train = pca.fit_transform(X["training"])

    out_dirs = {
        "training": EMBEDDINGS_DATA_DIR / "training",
        "validation": EMBEDDINGS_DATA_DIR / "validation",
        "test": EMBEDDINGS_DATA_DIR / "test",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    save_latents(Z_pca_train, idx["training"], out_dirs["training"] / "latent_pca.csv", "pca")
    save_latents(pca.transform(X["validation"]), idx["validation"], out_dirs["validation"] / "latent_pca.csv", "pca")
    save_latents(pca.transform(X["test"]), idx["test"], out_dirs["test"] / "latent_pca.csv", "pca")

    if pca_only:
        logger.info("Embeddings PCA listos (train/val/test).")
        return

    # 3) Autoencoder: cargar y encode en todos los splits
    model_path = MODELS_DIR / model_name
    logger.info(f"Cargando AE desde: {model_path}")
    model = load_autoencoder(model_path, input_dim=X["training"].shape[1])

    logger.info("Generando embeddings AE (train/val/test) ...")
    Z_ae_train = encode_in_batches(model, X["training"], batch_size=batch_size)
    Z_ae_val = encode_in_batches(model, X["validation"], batch_size=batch_size)
    Z_ae_test = encode_in_batches(model, X["test"], batch_size=batch_size)

    save_latents(Z_ae_train, idx["training"], out_dirs["training"] / "latent_ae.csv", "ae")
    save_latents(Z_ae_val, idx["validation"], out_dirs["validation"] / "latent_ae.csv", "ae")
    save_latents(Z_ae_test, idx["test"], out_dirs["test"] / "latent_ae.csv", "ae")

    logger.info(f"✨ Embeddings PCA+AE guardados en {EMBEDDINGS_DATA_DIR}")


if __name__ == "__main__":
    app()