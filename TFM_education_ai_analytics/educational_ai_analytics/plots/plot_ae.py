import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import typer
from sklearn.decomposition import PCA
import tensorflow as tf

from educational_ai_analytics.config import REPORTS_DIR, EMBEDDINGS_DATA_DIR, MODELS_DIR, FEATURES_DATA_DIR, AE_REPORTS_DIR
from educational_ai_analytics.modeling import StudentProfileAutoencoder
from .style import set_style

app = typer.Typer(help="Visualizaciones para el Autoencoder.")
set_style()

@app.command()
def loss_curve(
    history_path: Path = REPORTS_DIR / "ae_training_history.csv",
    output_name: str = "ae_loss_curve.png"
):
    """Genera la gráfica de pérdida (Loss) del Autoencoder."""
    if not history_path.exists():
        logger.error(f"No se encuentra el historial en {history_path}.")
        return

    logger.info(f"Graficando curva de pérdida de: {history_path}")
    df = pd.read_csv(history_path)
    
    plt.figure()
    plt.plot(df['loss'], label='Entrenamiento (Train)', linewidth=2, color='#3498db')
    plt.plot(df['val_loss'], label='Validación (Val)', linewidth=2, linestyle='--', color='#e74c3c')
    
    plt.title("Evolución de la Pérdida del Autoencoder")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (Mean Squared Error)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = AE_REPORTS_DIR / output_name
    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    logger.success(f"📈 Gráfica guardada en: {out_file}")

@app.command()
def latent_space(
    embeddings_file: Path = EMBEDDINGS_DATA_DIR / "training" / "latent_ae.csv",
    output_name: str = "latent_space.png",
    method: str = "pca",
    max_points: int = 30000,
    seed: int = 42,
    alpha: float = 0.35,
    point_size: int = 6,
):
    """Visualiza embeddings en 2D."""
    if not embeddings_file.exists():
        logger.error(f"No existen embeddings en {embeddings_file}.")
        return

    logger.info(f"Cargando embeddings: {embeddings_file}")
    df = pd.read_csv(embeddings_file, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(df) > max_points:
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(df.index.values, size=max_points, replace=False)
        df = df.loc[sample_idx]

    X = df.values.astype(np.float32)

    if method.lower() == "first2" and df.shape[1] >= 2:
        Z = X[:, :2]
        xlab, ylab = df.columns[0], df.columns[1]
    else:
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(X)
        xlab, ylab = "PC1", "PC2"

    plt.figure(figsize=(9, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=alpha, c="#3498db")
    plt.title(f"Espacio Latente | {embeddings_file.stem}")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.2)
    
    out_file = AE_REPORTS_DIR / output_name
    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.success(f"✨ Visualización guardada en: {out_file}")

@app.command()
def ae_reconstruction(
    features_file: Path = FEATURES_DATA_DIR / "training" / "engineered_features.csv",
    model_name: str = "best_static_autoencoder.keras",
    output_name: str = "ae_reconstruction_flow.png",
    max_points: int = 10000,
    seed: int = 42,
):
    """Visualiza el flujo completo del AE: Original -> Latente -> Recon."""
    if not features_file.exists():
        logger.error(f"No existen features en {features_file}")
        return

    df = pd.read_csv(features_file, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)
    
    X_in = df.values.astype(np.float32)
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        logger.error(f"No existe el modelo en {model_path}")
        return

    ae = tf.keras.models.load_model(
        model_path, 
        custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
        compile=False
    )
    
    Z_lat = ae.encode(X_in, training=False).numpy()
    X_rec = ae.decode(Z_lat, training=False).numpy()

    pca_lens = PCA(n_components=2, random_state=seed).fit(X_in)
    p_in = pca_lens.transform(X_in)
    p_rec = pca_lens.transform(X_rec)
    p_lat = PCA(n_components=2, random_state=seed).fit_transform(Z_lat)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    
    axes[0].scatter(p_in[:, 0], p_in[:, 1], s=3, alpha=0.3, color="gray")
    axes[0].set_title("1. Datos Originales")
    
    axes[1].scatter(p_lat[:, 0], p_lat[:, 1], s=3, alpha=0.4, color="crimson")
    axes[1].set_title(f"2. Espacio Latente ({Z_lat.shape[1]}D)")
    
    axes[2].scatter(p_rec[:, 0], p_rec[:, 1], s=3, alpha=0.3, color="royalblue")
    axes[2].set_title("3. Reconstrucción")

    for ax in axes:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.1)

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(AE_REPORTS_DIR / output_name, dpi=200)
    plt.close()
    logger.success(f"✨ Flujo de reconstrucción guardado en: {AE_REPORTS_DIR / output_name}")

if __name__ == "__main__":
    app()
