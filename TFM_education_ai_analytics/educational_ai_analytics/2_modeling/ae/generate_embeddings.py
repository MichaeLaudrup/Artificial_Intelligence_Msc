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
from sklearn.decomposition import PCA

from educational_ai_analytics.config import (
    FEATURES_DATA_DIR,
    EMBEDDINGS_DATA_DIR,
    MODELS_DIR,
    W_WINDOWS,
)
from .hyperparams import AE_PARAMS
from .autoencoder import StudentProfileAutoencoder

app = typer.Typer()

def _set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
    return X, df.index

def save_csv(data: np.ndarray, index: pd.Index, out_path: Path, prefix: str):
    cols = [f"{prefix}_{i:02d}" for i in range(data.shape[1])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=index, columns=cols).to_csv(out_path)
    logger.info(f"💾 Guardado: {out_path} | shape={data.shape}")

@app.command()
def main(
    seed: int = 42,
    batch_size: int = 1024,
    use_global_model: bool = True,
):
    """
    Genera embeddings (PCA y AE Latent) para las ventanas configuradas utilizando una estrategia GLOBAL.
    Tanto el PCA como el AE se aplican de forma alineada (entrenados con todas las semanas apiladas).
    """
    _set_seed(seed)
    
    windows = sorted([int(w) for w in W_WINDOWS])
    logger.info(f"🧬 Generando embeddings (PCA + AE) de forma GLOBAL para windows: {windows}")

    # 1) Carga de todos los datos en memoria para el PCA Global
    logger.info("📂 Cargando datos de todas las ventanas para inicializar PCA Global...")
    data_all_windows = {}
    train_stacks = []

    for W in windows:
        data_all_windows[W] = {}
        for split in ["training", "validation", "test"]:
            path = FEATURES_DATA_DIR / split
            if path.exists():
                try:
                    X, idx = load_features(path, W=W)
                    data_all_windows[W][split] = (X, idx)
                    if split == "training":
                        train_stacks.append(X)
                except Exception as e:
                    logger.warning(f"   ⚠️ Error cargando W{W:02d} split {split}: {e}")

    if not train_stacks:
        logger.error("❌ No se encontraron datos de entrenamiento para inicializar el PCA.")
        return

    # 2) Ajuste de PCA Global (con todas las semanas apiladas)
    logger.info("📉 Ajustando PCA GLOBAL (apilando todas las semanas)...")
    X_train_full = np.vstack(train_stacks)
    pca_global = PCA(n_components=AE_PARAMS.latent_dim, random_state=seed)
    pca_global.fit(X_train_full)
    logger.success(f"✅ PCA Global ajustado con {X_train_full.shape} registros.")

    # 3) Carga del Modelo Autoencoder Único (Global)
    global_model_path = MODELS_DIR / "ae_best_global.keras"
    ae_global = None
    if use_global_model:
        if global_model_path.exists():
            logger.info(f"🧠 Cargando MODELO AE GLOBAL: {global_model_path}")
            ae_global = tf.keras.models.load_model(
                global_model_path,
                custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
                compile=False
            )
        else:
            logger.warning(f"⚠️ No se encontró modelo AE global en {global_model_path}. Se buscarán modelos específicos.")

    # 4) Transformación y Guardado
    for W in windows:
        logger.info(f"\n🚀 PROCESANDO VENTANA: upto_w{W:02d}")
        
        if not data_all_windows[W]:
            continue

        # PCA Latent (usando el global ajustado arriba)
        logger.info(f"   📉 Aplicando PCA Global...")
        for split_name, (X, idx) in data_all_windows[W].items():
            out_dir = EMBEDDINGS_DATA_DIR / split_name / f"upto_w{W:02d}"
            Z_pca = pca_global.transform(X)
            save_csv(Z_pca, idx, out_dir / "pca_latent.csv", "pca")

        # Autoencoder Latent
        ae_to_use = ae_global
        if ae_to_use is None:
            # Fallback a modelo específico de W si no hay global
            model_w_path = MODELS_DIR / f"ae_best_w{W:02d}.keras"
            if model_w_path.exists():
                logger.info(f"   🧠 Cargando fallback AE para W{W:02d}...")
                ae_to_use = tf.keras.models.load_model(
                    model_w_path,
                    custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
                    compile=False
                )

        if ae_to_use is not None:
            logger.info(f"   🧠 Generando AE Latent y DEC Clusters...")
            for split_name, (X, idx) in data_all_windows[W].items():
                out_dir = EMBEDDINGS_DATA_DIR / split_name / f"upto_w{W:02d}"
                # Get both embeddings and cluster assignments
                # StudentProfileAutoencoder.call returns (x_recon, q)
                # But we want z too. We can use model(X) or encode(X) + clustering_layer(z)
                z = ae_to_use.get_embeddings(X, batch_size=batch_size).numpy()
                q = ae_to_use.clustering_layer(z).numpy()
                
                # Save embeddings
                save_csv(z, idx, out_dir / "ae_latent.csv", "z")
                
                # Save DEC Segmentation
                cluster_id = q.argmax(axis=1)
                seg_df = pd.DataFrame(index=idx)
                seg_df["cluster_id"] = cluster_id
                for j in range(q.shape[1]):
                    seg_df[f"p_cluster_{j}"] = q[:, j]
                
                seg_path = out_dir / "segmentation_dec.csv"
                seg_df.to_csv(seg_path)
                logger.info(f"   📍 Saved DEC Segmentation: {seg_path}")
        else:
            logger.warning(f"   ⚠️ No se encontró modelo AE para W{W:02d}")

    logger.success("✨ Proceso de embeddings GLOBAL completado.")

if __name__ == "__main__":
    app()