import os
import warnings

# Silenciar warnings de Protobuf y logs de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from educational_ai_analytics.config import (
    REPORTS_DIR,
    EMBEDDINGS_DATA_DIR,
    MODELS_DIR,
    FEATURES_DATA_DIR,
    AE_REPORTS_DIR,
    W_WINDOWS,
)

# Import dinámico para evitar SyntaxError por el nombre de carpeta "2_modeling"
import importlib

_mod_modeling = importlib.import_module("educational_ai_analytics.2_modeling")
StudentProfileAutoencoder = _mod_modeling.StudentProfileAutoencoder

# UMAP (opcional)
try:
    import umap  # pip install umap-learn
except Exception:
    umap = None

from .style import set_style

set_style()

logger.info(f"📊 Configured W_WINDOWS: {W_WINDOWS}")

app = typer.Typer(help="Visualizaciones para el Autoencoder (multi-W).")


# ---------- helpers ----------
def _history_path_for_W(W: int) -> Path:
    hist_dir = REPORTS_DIR / "ae" / ".history"
    global_path = hist_dir / "ae_training_history_global.csv"
    if global_path.exists():
        return global_path
    return hist_dir / f"ae_training_history_w{int(W):02d}.csv"


def _model_path_for_W(W: int) -> Path:
    global_path = MODELS_DIR / "ae_best_global.keras"
    if global_path.exists():
        return global_path
    return MODELS_DIR / f"ae_best_w{int(W):02d}.keras"


def _embeddings_path_for_W(W: int, split: str, emb_type: str = "ae") -> Path:
    split = split.lower().strip()
    # Mapeo de nombres cortos a nombres de carpeta si es necesario
    split_map = {"train": "training", "val": "validation", "test": "test"}
    folder_split = split_map.get(split, split)

    if folder_split not in {"training", "validation", "test"}:
        raise ValueError("split debe ser 'training', 'validation' o 'test'")

    # Estructura: 4_embeddings/{split}/upto_w{W}/{ae|pca}_latent.csv
    fname = f"{emb_type}_latent.csv"
    return EMBEDDINGS_DATA_DIR / folder_split / f"upto_w{int(W):02d}" / fname


def load_features_for_W(split_dir: Path, W: int):
    """
    Carga y concatena features para el AE (Static + Dynamic UptoW).
    """
    X0_path = split_dir / "day0_static_features.csv"
    Xdyn_path = split_dir / "ae_uptow_features" / f"ae_uptow_features_w{int(W):02d}.csv"

    if not X0_path.exists():
        raise FileNotFoundError(f"Falta {X0_path}")
    if not Xdyn_path.exists():
        raise FileNotFoundError(f"Falta {Xdyn_path}")

    df0 = pd.read_csv(X0_path, index_col=0)
    dfdyn = pd.read_csv(Xdyn_path, index_col=0)

    df = pd.concat([df0, dfdyn.reindex(df0.index)], axis=1).fillna(0.0)
    X = df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
    return X, df.index, df


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _project(
    X: np.ndarray,
    method: str,
    seed: int,
    n_components: int = 2,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
):
    """
    Proyecta X a n_components usando PCA/UMAP/t-SNE/first2.

    Devuelve:
      - (Z, lab1, lab2) si n_components==2
      - (Z, lab1, lab2, lab3) si n_components==3
    """
    method = method.lower().strip()

    if method == "first2":
        if X.shape[1] < n_components:
            raise ValueError(f"first2 requiere al menos {n_components} dims en X")
        Z = X[:, :n_components]
        labs = [f"dim{i+1}" for i in range(n_components)]
        return (Z, *labs)

    if method == "pca":
        Z = PCA(n_components=n_components, random_state=seed).fit_transform(X)
        labs = [f"PC{i+1}" for i in range(n_components)]
        return (Z, *labs)

    if method == "umap":
        if umap is None:
            raise ImportError("UMAP no está instalado. Ejecuta: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=seed,
        )
        Z = reducer.fit_transform(X)
        labs = [f"UMAP{i+1}" for i in range(n_components)]
        return (Z, *labs)

    if method == "tsne":
        Z = TSNE(
            n_components=n_components,
            random_state=seed,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            init="pca",
        ).fit_transform(X)
        labs = [f"tSNE{i+1}" for i in range(n_components)]
        return (Z, *labs)

    raise ValueError("method debe ser: 'pca', 'umap', 'tsne', 'first2'")


# ---------- commands ----------
@app.command()
def loss_curve(
    W: int = W_WINDOWS[0],
    output_name: str = "",
):
    """
    Curva de pérdida (loss/val_loss) para una W concreta.
    Lee: reports/ae/ae_training_history_wXX.csv
    """
    history_path = _history_path_for_W(W)
    if not history_path.exists():
        logger.warning(f"   ⚠️ Saltando curva de pérdida: No se encuentra historial en {history_path}.")
        return

    df = pd.read_csv(history_path)

    plt.figure(figsize=(9, 5))
    if "loss" in df.columns:
        plt.plot(df["loss"], label="Train loss", linewidth=2, color="#3498db")
    if "val_loss" in df.columns:
        plt.plot(df["val_loss"], label="Val loss", linewidth=2, linestyle="--", color="#e74c3c")

    plt.title(f"Evolución de la pérdida del AE | {'Global' if 'global' in str(history_path) else f'W{int(W):02d}'}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (Huber)" if "loss" in df.columns else "Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = AE_REPORTS_DIR / (output_name or f"ae_loss_curve_w{int(W):02d}.png")
    plt.savefig(out_file, bbox_inches="tight", dpi=200)
    plt.close()
    logger.success(f"📈 Gráfica guardada en: {out_file}")


@app.command()
def latent_space(
    W: int = W_WINDOWS[0],
    split: str = "train",
    emb_type: str = "ae",
    output_name: str = "",
    method: str = "pca",
    max_points: int = 30000,
    seed: int = 42,
    alpha: float = 0.35,
    point_size: int = 6,
    # UMAP params
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    # t-SNE params
    tsne_perplexity: float = 30.0,
):
    """
    Visualiza embeddings en 2D para W y split.
    emb_type: 'ae' o 'pca'
    method: 'pca' | 'umap' | 'tsne' | 'first2'
    """
    embeddings_file = _embeddings_path_for_W(W, split, emb_type=emb_type)
    if not embeddings_file.exists():
        logger.error(f"No existen embeddings en {embeddings_file}.")
        return

    logger.info(f"Cargando embeddings: {embeddings_file}")
    df = _safe_read_csv(embeddings_file)

    if len(df) > max_points:
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(df.index.values, size=max_points, replace=False)
        df = df.loc[sample_idx]

    X = df.values.astype(np.float32)

    try:
        Z, xlab, ylab = _project(
            X,
            method=method,
            seed=seed,
            n_components=2,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            tsne_perplexity=tsne_perplexity,
        )
    except Exception as e:
        logger.error(f"Error proyectando ({method}): {e}")
        return

    plt.figure(figsize=(9, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=alpha, c="#3498db")
    plt.title(f"Espacio latente ({method.upper()}) | W{int(W):02d} | {split} | {emb_type.upper()}")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.2)

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = AE_REPORTS_DIR / (output_name or f"latent_space_{method}_w{int(W):02d}_{split}_{emb_type}.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.success(f"✨ Visualización guardada en: {out_file}")


@app.command()
def ae_reconstruction(
    W: int = W_WINDOWS[0],
    split: str = "training",
    output_name: str = "",
    max_points: int = 10000,
    seed: int = 42,
):
    """
    Visualiza el flujo del AE: Original -> Latente -> Recon.
    - Usa features: day0_static_features.csv + ae_uptow_features_wXX.csv
    - Usa modelo: models/ae_best_wXX.keras

    Nota: aquí mantenemos PCA como "lens" para comparar original vs reconstrucción con la MISMA proyección.
    """
    split = split.lower().strip()
    if split not in {"training", "validation", "test"}:
        logger.error("split debe ser 'training', 'validation' o 'test'")
        return

    split_dir = FEATURES_DATA_DIR / split
    if not split_dir.exists():
        logger.error(f"No existe split_dir: {split_dir}")
        return

    model_path = _model_path_for_W(W)
    if not model_path.exists():
        logger.error(f"No existe el modelo en {model_path}")
        return

    # cargar features (concat)
    try:
        X_in, idx, df = load_features_for_W(split_dir, W=W)
    except FileNotFoundError as e:
        logger.warning(f"   ⚠️ Saltando reconstrucción para W{W:02d}: {e}")
        return

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)
        X_in = df.values.astype(np.float32)

    # cargar modelo
    ae = tf.keras.models.load_model(
        model_path,
        custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
        compile=False,
    )

    # encode/decode
    Z_lat = ae.encode(X_in, training=False).numpy()
    X_rec = ae.decode(Z_lat, training=False).numpy()

    # PCA lens (mismo lens para comparar original vs recon)
    # PCA 2D
    pca_lens_2d = PCA(n_components=2, random_state=seed).fit(X_in)
    p_in_2d = pca_lens_2d.transform(X_in)
    p_rec_2d = pca_lens_2d.transform(X_rec)
    p_lat_2d = PCA(n_components=2, random_state=seed).fit_transform(Z_lat)

    # PCA 3D
    pca_lens_3d = PCA(n_components=3, random_state=seed).fit(X_in)
    p_in_3d = pca_lens_3d.transform(X_in)
    p_rec_3d = pca_lens_3d.transform(X_rec)
    p_lat_3d = PCA(n_components=3, random_state=seed).fit_transform(Z_lat)

    # Figure with 2 rows: Row 0 -> 2D, Row 1 -> 3D
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)

    titles = [
        "1) Datos originales (PCA lens)",
        f"2) Espacio latente (PCA) | z_dim={Z_lat.shape[1]}",
        "3) Reconstrucción (PCA lens original)",
    ]
    colors = ["gray", "crimson", "royalblue"]

    # Fila 1: 2D
    axes_2d = [
        fig.add_subplot(2, 3, 1),
        fig.add_subplot(2, 3, 2),
        fig.add_subplot(2, 3, 3),
    ]
    data_2d = [p_in_2d, p_lat_2d, p_rec_2d]

    for ax, data, title, color in zip(axes_2d, data_2d, titles, colors):
        ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.3, color=color)
        ax.set_title(f"{title} [2D]")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.12)

    # Fila 2: 3D
    axes_3d = [
        fig.add_subplot(2, 3, 4, projection="3d"),
        fig.add_subplot(2, 3, 5, projection="3d"),
        fig.add_subplot(2, 3, 6, projection="3d"),
    ]
    data_3d = [p_in_3d, p_lat_3d, p_rec_3d]

    for ax, data, title, color in zip(axes_3d, data_3d, titles, colors):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, alpha=0.25, color=color)
        ax.set_title(f"{title} [3D]")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = AE_REPORTS_DIR / (output_name or f"ae_reconstruction_flow_w{int(W):02d}_{split}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.success(f"✨ Flujo de reconstrucción guardado en: {out_file}")


@app.command()
def compare_latent_spaces(
    W_list: str = ",".join(map(str, W_WINDOWS)),
    split: str = "train",
    emb_type: str = "ae",
    method: str = "pca",
    max_points: int = 15000,
    seed: int = 42,
    alpha: float = 0.25,
    point_size: int = 4,
    # UMAP params
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    # t-SNE params
    tsne_perplexity: float = 30.0,
):
    """
    Compara visualmente varios espacios latentes (cada W en su subplot).
    method: 'pca' | 'umap' | 'tsne' | 'first2'
    """
    split = split.lower().strip()
    if split not in {"train", "val", "test"}:
        logger.error("split debe ser 'train', 'val' o 'test'")
        return

    Ws = [int(x.strip()) for x in W_list.split(",") if x.strip()]
    if not Ws:
        logger.error("W_list vacío.")
        return

    n = len(Ws)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 7 * nrows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    last_i = -1
    for i, W in enumerate(Ws):
        last_i = i
        ax = axes[i]
        emb_path = _embeddings_path_for_W(W, split, emb_type=emb_type)
        if not emb_path.exists():
            ax.set_title(f"W{W:02d} (missing)")
            ax.axis("off")
            continue

        df = _safe_read_csv(emb_path)
        if len(df) > max_points:
            rng = np.random.RandomState(seed)
            sample_idx = rng.choice(df.index.values, size=max_points, replace=False)
            df = df.loc[sample_idx]

        X = df.values.astype(np.float32)

        try:
            Z, xlab, ylab = _project(
                X,
                method=method,
                seed=seed,
                n_components=2,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                umap_metric=umap_metric,
                tsne_perplexity=tsne_perplexity,
            )
        except Exception as e:
            ax.set_title(f"W{W:02d} ({method}) error")
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
            ax.axis("off")
            continue

        ax.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=alpha, c="#3498db")
        ax.set_title(f"{method.upper()} ({emb_type.upper()}) | W{W:02d} | {split}")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.2)

    # apagar axes sobrantes
    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = AE_REPORTS_DIR / f"latent_spaces_compare_{method}_{split}_{emb_type}_W{W_list.replace(',', '-')}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.success(f"🧩 Comparativa guardada en: {out_file}")


@app.command()
def compare_pca_vs_ae(
    W_list: str = ",".join(map(str, W_WINDOWS)),
    split: str = "train",
    method: str = "pca",
    max_points: int = 15000,
    seed: int = 42,
    alpha: float = 0.25,
    point_size: int = 4,
    # UMAP params
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    # t-SNE params
    tsne_perplexity: float = 30.0,
):
    """
    Compara visualmente PCA_latent vs AE_latent para varias W.
    Grid: Rows = W, Columns = {PCA_latent, AE_latent}
    method controla el "lente" (pca/umap/tsne/first2) para proyectar a 2D.
    """
    split = split.lower().strip()

    logger.info(f"🧬 compare_pca_vs_ae: W_list received='{W_list}'")

    Ws = [int(x.strip()) for x in W_list.split(",") if x.strip()]
    Ws = [w for w in Ws if w in W_WINDOWS]

    if not Ws:
        logger.warning("⚠️ W_list resultó vacía tras filtrar con W_WINDOWS.")
        return

    nrows = len(Ws)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), constrained_layout=True)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, W in enumerate(Ws):
        for j, emb_type in enumerate(["pca", "ae"]):
            ax = axes[i, j]
            emb_path = _embeddings_path_for_W(W, split, emb_type=emb_type)

            if not emb_path.exists():
                ax.set_title(f"W{W:02d} {emb_type.upper()} (missing)")
                ax.axis("off")
                continue

            df = _safe_read_csv(emb_path)
            if len(df) > max_points:
                rng = np.random.RandomState(seed)
                sample_idx = rng.choice(df.index.values, size=max_points, replace=False)
                df = df.loc[sample_idx]

            X = df.values.astype(np.float32)

            try:
                Z, xlab, ylab = _project(
                    X,
                    method=method,
                    seed=seed,
                    n_components=2,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                    umap_metric=umap_metric,
                    tsne_perplexity=tsne_perplexity,
                )
            except Exception as e:
                ax.set_title(f"W{W:02d} | {emb_type.upper()} | {method.upper()} error")
                ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)
                ax.axis("off")
                continue

            color = "#3498db" if emb_type == "pca" else "#e74c3c"
            ax.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=alpha, c=color)
            ax.set_title(f"W{W:02d} | {emb_type.upper()} | lens={method.upper()}")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.15)

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = AE_REPORTS_DIR / f"compare_pca_vs_ae_lens{method}_{split}_W{W_list.replace(',', '-')}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    logger.success(f"🖼️ Grid comparativo PCA_latent vs AE_latent guardado en: {out_file}")


@app.command()
def clean():
    """
    Limpia el directorio de reportes, moviendo historiales a .history y borrando lo demás.
    """
    if AE_REPORTS_DIR.exists():
        logger.info(f"🧹 Limpiando y organizando reportes en: {AE_REPORTS_DIR}")
        hist_dir = AE_REPORTS_DIR / ".history"
        hist_dir.mkdir(parents=True, exist_ok=True)

        for item in AE_REPORTS_DIR.iterdir():
            if item.name == ".history":
                continue

            if item.is_file():
                if item.suffix == ".csv":
                    target = hist_dir / item.name
                    if not target.exists():
                        import shutil

                        shutil.move(str(item), str(target))
                    else:
                        item.unlink()
                else:
                    item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)

        logger.success("✅ Directorio de reportes limpio.")


if __name__ == "__main__":
    app()
