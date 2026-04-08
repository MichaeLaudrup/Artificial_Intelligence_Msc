import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import typer
from sklearn.decomposition import PCA

from educational_ai_analytics.config import FIGURES_DIR, REPORTS_DIR, EMBEDDINGS_DATA_DIR

app = typer.Typer(help="Herramientas de visualización para el TFM.")


plt.style.use('ggplot') 
sns.set_palette("viridis")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150
})

@app.command()
def loss_curve(
    history_path: Path = REPORTS_DIR / "ae_training_history.csv",
    output_name: str = "ae_loss_curve.png"
):
    """
    Genera la gráfica de pérdida (Loss) del Autoencoder para verificar convergencia.
    """
    if not history_path.exists():
        logger.error(f"No se encuentra el historial en {history_path}. Entrena el modelo primero.")
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
    
    out_file = FIGURES_DIR / output_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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
    """
    Visualiza embeddings en 2D y guarda la figura.
    - Si embeddings tiene >2 dims, method="pca" hace PCA->2D para plot (más honesto que pillar 2 columnas).
    - method="first2" usa las 2 primeras columnas tal cual (solo tiene sentido si ya son coords 2D).
    """
    if not embeddings_file.exists():
        logger.error(f"No se encuentran embeddings en {embeddings_file}. Ejecuta 'make embeddings' primero.")
        raise typer.Exit(code=1)

    logger.info(f"Cargando embeddings: {embeddings_file}")
    df = pd.read_csv(embeddings_file, index_col=0)

    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    
    if len(df) > max_points:
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(df.index.values, size=max_points, replace=False)
        df = df.loc[sample_idx].sort_index()
        logger.info(f"Sampling: usando {len(df)} puntos (max_points={max_points})")
    else:
        df = df.sort_index()
        logger.info(f"Usando todos los puntos: N={len(df)}")

    X = df.values.astype(np.float32)

    
    if method.lower() == "first2":
        if df.shape[1] < 2:
            logger.error("El embedding tiene menos de 2 columnas. No puedo hacer 'first2'.")
            raise typer.Exit(code=1)
        Z = X[:, :2]
        xlab, ylab = df.columns[0], df.columns[1]
        title = f"Espacio Latente (first2) | {embeddings_file.stem}"
    else:
        
        pca = PCA(n_components=2, random_state=seed)
        Z = pca.fit_transform(X)
        var = pca.explained_variance_ratio_.sum()
        xlab, ylab = "PC1", "PC2"
        title = f"Espacio Latente (PCA->2D) | {embeddings_file.stem} | var={var:.2%}"

    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIGURES_DIR / output_name

    plt.figure(figsize=(9, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=point_size, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    logger.success(f"✨ Visualización guardada en: {out_file}")


@app.command()
def ae_reconstruction(
    features_file: Path = Path("/workspace/TFM_education_ai_analytics/data/3_features/training/engineered_features.csv"),
    model_name: str = "best_static_autoencoder.keras",
    output_name: str = "ae_reconstruction_flow.png",
    max_points: int = 10000,
    seed: int = 42,
):
    """
    Visualiza el flujo completo del AE: Datos Originales -> Espacio Latente -> Reconstrucción.
    """
    import tensorflow as tf
    from educational_ai_analytics.modeling import StudentProfileAutoencoder
    from educational_ai_analytics.config import MODELS_DIR, FEATURES_DATA_DIR

    
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
    axes[0].set_title("1. Datos Originales (43D)")
    
    axes[1].scatter(p_lat[:, 0], p_lat[:, 1], s=3, alpha=0.4, color="crimson")
    axes[1].set_title(f"2. Espacio Latente ({Z_lat.shape[1]}D)\n(Compresión IA)")
    
    axes[2].scatter(p_rec[:, 0], p_rec[:, 1], s=3, alpha=0.3, color="royalblue")
    axes[2].set_title("3. Reconstrucción (43D)\n(Lo que el AE decodifica)")

    for ax in axes:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / output_name, dpi=200)
    plt.close()
    
    logger.success(f"✨ Flujo de reconstrucción guardado en: {FIGURES_DIR / output_name}")


if __name__ == "__main__":
    app()
