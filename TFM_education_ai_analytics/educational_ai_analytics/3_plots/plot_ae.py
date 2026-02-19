import os
import warnings

# Silenciar warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import typer
from sklearn.decomposition import PCA
import tensorflow as tf

from educational_ai_analytics.config import (
    REPORTS_DIR,
    EMBEDDINGS_DATA_DIR,
    AE_MODELS_DIR,
    MODELS_DIR,
    FEATURES_DATA_DIR,
    AE_REPORTS_DIR,
    W_WINDOWS,
)

# Import del modelo
import importlib
_mod_modeling = importlib.import_module("educational_ai_analytics.2_modeling")
StudentProfileAutoencoder = _mod_modeling.StudentProfileAutoencoder

from .style import set_style
set_style()

app = typer.Typer(help="Visualizaciones para el Autoencoder Global.")


def _ae_model_path() -> Path:
    p = AE_MODELS_DIR / "ae_best_global.keras"
    if p.exists():
        return p
    return MODELS_DIR / "ae_best_global.keras"

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _embeddings_path(W: int, split: str, emb_type: str = "ae") -> Path:
    split_map = {"train": "training", "val": "validation", "test": "test"}
    folder_split = split_map.get(split, split)
    return EMBEDDINGS_DATA_DIR / folder_split / f"upto_w{int(W):02d}" / f"{emb_type}_latent.csv"

def load_features_for_W(split_dir: Path, W: int):
    X0_path = split_dir / "day0_static_features.csv"
    Xdyn_path = split_dir / "ae_uptow_features" / f"ae_uptow_features_w{int(W):02d}.csv"
    df0 = pd.read_csv(X0_path, index_col=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    dfdyn = pd.read_csv(Xdyn_path, index_col=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df = pd.concat([df0, dfdyn.reindex(df0.index)], axis=1).fillna(0.0)
    return df.values.astype(np.float32), df.index, df

def _load_clusters(W: int, split: str) -> pd.Series:
    """Intenta cargar cluster_id de DEC o GMM."""
    split_map = {"train": "training", "val": "validation", "test": "test"}
    s = split_map.get(split, split)
    base = EMBEDDINGS_DATA_DIR / s / f"upto_w{int(W):02d}"
    
    # Prioridad 1: DEC (nuestra nueva estrategia)
    dec_path = base / "segmentation_dec.csv"
    if dec_path.exists():
        return pd.read_csv(dec_path, index_col=0)["cluster_id"]
    
    # Prioridad 2: GMM
    gmm_path = base / "segmentation_gmm_ae.csv"
    if gmm_path.exists():
        return pd.read_csv(gmm_path, index_col=0)["cluster_id"]
    
    return pd.Series()

@app.command()
def loss_curve():
    """Curva de pérdida del entrenamiento Global."""
    history_path = REPORTS_DIR / "ae" / ".history" / "ae_training_history_global.csv"
    if not history_path.exists():
        logger.warning(f"⚠️ No hay historial en {history_path}")
        return

    df = pd.read_csv(history_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["loss"], label="Train loss", color="#3498db")
    plt.plot(df["val_loss"], label="Val loss", color="#e74c3c", linestyle="--")
    plt.title("Evolución de la pérdida (AE Global Concat)")
    plt.xlabel("Épocas"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    
    out = AE_REPORTS_DIR / "ae_loss_curve_global.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    logger.success(f"📈 Guardado: {out}")

@app.command()
def ae_reconstruction(split: str = "training"):
    """Visualiza el flujo del AE para todas las semanas: Original -> Latente -> Recon."""
    split_dir = FEATURES_DATA_DIR / split
    model_path = _ae_model_path()
    if not model_path.exists(): 
        logger.error(f"❌ No se encuentra el modelo en {model_path}")
        return

    logger.info(f"🎨 Cargando modelo para reconstrucción: {model_path}")
    ae = tf.keras.models.load_model(
        model_path, 
        custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder}, 
        compile=False
    )

    windows = sorted([int(w) for w in W_WINDOWS])
    n_rows = len(windows)
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0) # Garantizar 2D axes[row, col]

    titles = ["1) Original (PCA Lens)", "2) Espacio Latente", "3) Reconstrucción (PCA Lens)"]
    cols = ["#7f8c8d", "#e74c3c", "#3498db"]

    for i, W in enumerate(windows):
        logger.info(f"   ⏳ Procesando Semana {W}...")
        try:
            X_in, idx, _ = load_features_for_W(split_dir, W)
        except Exception as e:
            logger.warning(f"   ⚠️ Error cargando W{W}: {e}")
            continue

        # Pasar por el AE
        Z_lat = ae.encode(X_in, training=False).numpy()
        X_rec = ae.decode(Z_lat, training=False).numpy()

        # Proyecciones 2D para visualización
        pca_in = PCA(n_components=2, random_state=42).fit(X_in)
        p_in = pca_in.transform(X_in)
        p_rec = pca_in.transform(X_rec)
        
        pca_lat = PCA(n_components=2, random_state=42)
        p_lat = pca_lat.fit_transform(Z_lat)

        # Cargar clusters para colorear
        clusters = _load_clusters(W, split)
        c_vals = clusters.reindex(idx).values if not clusters.empty else None
        cmap = "viridis" if c_vals is not None else None

        data_row = [p_in, p_lat, p_rec]

        for j, (d, t, col_default) in enumerate(zip(data_row, titles, cols)):
            ax = axes[i, j]
            if c_vals is not None:
                ax.scatter(d[:, 0], d[:, 1], s=3, alpha=0.3, c=c_vals, cmap=cmap, edgecolors='none')
            else:
                ax.scatter(d[:, 0], d[:, 1], s=3, alpha=0.25, color=col_default, edgecolors='none')
            
            if i == 0:
                ax.set_title(t, fontsize=12, fontweight='bold', pad=15)
            if j == 0:
                ax.set_ylabel(f"W{W:02d}", fontsize=14, fontweight='bold', labelpad=15)
            
            ax.grid(True, alpha=0.1, linestyle='--')
            ax.set_xticks([]); ax.set_yticks([])

    out = AE_REPORTS_DIR / f"ae_reconstruction_multi_row_{split}.png"
    plt.savefig(out, dpi=180); plt.close()
    logger.success(f"✨ Dashboard de reconstrucción guardado: {out}")

@app.command()
def latent_space(W: int = W_WINDOWS[0], split: str = "train", method: str = "pca"):
    """Visualiza el espacio latente del AE en 2D coloreado por clústeres."""
    path = _embeddings_path(W, split, "ae")
    df = _safe_read_csv(path)
    if df.empty: return

    X = df.values.astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    # Cargar clusters
    clusters = _load_clusters(W, split)
    c_vals = clusters.reindex(df.index).values if not clusters.empty else None

    plt.figure(figsize=(8, 6))
    if c_vals is not None:
        scatter = plt.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.4, c=c_vals, cmap="viridis")
        plt.colorbar(scatter, label="Cluster ID")
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.3, color="#3498db")
    
    # Intentar mostrar centros de DEC
    model_path = _ae_model_path()
    if model_path.exists():
        try:
            ae = tf.keras.models.load_model(
                model_path, 
                custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder}, 
                compile=False
            )
            # Centros están en ae.clustering_layer.clusters
            centers = ae.get_layer("clustering_output").get_weights()[0]
            c_2d = pca.transform(centers)
            plt.scatter(c_2d[:, 0], c_2d[:, 1], s=100, c='red', marker='X', edgecolors='white', label='DEC Centers', zorder=10)
            plt.legend()
        except Exception as e:
            logger.warning(f"No se pudieron proyectar los centros DEC: {e}")

    plt.title(f"AE Latent Space & DEC Clusters | W{W} | {split}")
    plt.grid(True, alpha=0.1)
    
    out = AE_REPORTS_DIR / f"latent_space_w{W:02d}_{split}.png"
    plt.savefig(out, dpi=180); plt.close()
    logger.success(f"✨ Guardado: {out}")

@app.command()
def compare_pca_vs_ae(split: str = "train", method: str = "pca"):
    """Compara PCA vs AE para todas las ventanas."""
    Ws = sorted(W_WINDOWS)
    nrows = len(Ws); ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), constrained_layout=True)
    if nrows == 1: axes = np.expand_dims(axes, axis=0)

    for i, W in enumerate(Ws):
        for j, etype in enumerate(["pca", "ae"]):
            ax = axes[i, j]
            path = _embeddings_path(W, split, etype)
            df = _safe_read_csv(path)
            if df.empty: continue
            
            Z = PCA(n_components=2, random_state=42).fit_transform(df.values)
            color = "#3498db" if j==0 else "#e74c3c"
            ax.scatter(Z[:,0], Z[:,1], s=2, alpha=0.2, color=color)
            ax.set_title(f"W{W} | {etype.upper()}"); ax.grid(True, alpha=0.1)

    out = AE_REPORTS_DIR / f"compare_pca_vs_ae_{split}.png"
    plt.savefig(out, dpi=150); plt.close()
    logger.success(f"🖼️ Guardado: {out}")

@app.command()
def compare_latent_spaces(split: str = "train", method: str = "pca"):
    """Compara espacios latentes de AE entre ventanas."""
    Ws = sorted(W_WINDOWS)
    fig, axes = plt.subplots(1, len(Ws), figsize=(5 * len(Ws), 4), constrained_layout=True)
    if len(Ws) == 1: axes = [axes]

    for i, W in enumerate(Ws):
        ax = axes[i]
        path = _embeddings_path(W, split, "ae")
        df = _safe_read_csv(path)
        if df.empty: continue
        Z = PCA(n_components=2, random_state=42).fit_transform(df.values)
        ax.scatter(Z[:,0], Z[:,1], s=2, alpha=0.3, color="#2ecc71")
        ax.set_title(f"W{W} Latent Space"); ax.grid(True, alpha=0.1)

    out = AE_REPORTS_DIR / f"compare_latent_spaces_{split}.png"
    plt.savefig(out, dpi=150); plt.close()
    logger.success(f"🧩 Guardado: {out}")

@app.command()
def clean():
    """Limpia reportes."""
    if AE_REPORTS_DIR.exists():
        import shutil
        hist_dir = AE_REPORTS_DIR / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        for item in AE_REPORTS_DIR.iterdir():
            if item.name == ".history": continue
            if item.is_file() and item.suffix == ".csv":
                shutil.move(str(item), str(hist_dir / item.name))
            elif item.is_file(): item.unlink()
        logger.success("✅ Limpio.")

if __name__ == "__main__":
    app()
