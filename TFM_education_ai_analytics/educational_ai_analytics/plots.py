import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import typer

from educational_ai_analytics.config import FIGURES_DIR, REPORTS_DIR, EMBEDDINGS_DATA_DIR

app = typer.Typer(help="Herramientas de visualización para el TFM.")

# Configuración estética global (Modern & Academic)
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
    output_name: str = "latent_space_ae.png"
):
    """
    Visualiza el espacio latente (2D) si AE o PCA ya han sido ejecutados.
    """
    if not embeddings_file.exists():
        logger.error(f"No se encuentran embeddings en {embeddings_file}. Ejecuta 'make embeddings' primero.")
        return

    logger.info("Generando visualización del espacio latente...")
    df = pd.read_csv(embeddings_file, index_col=0)
    
    # Si tenemos más de 2 dimensiones, usamos el nombre de las columnas o tomamos las 2 primeras
    plt.figure()
    sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df, alpha=0.5, color='#2ecc71')
    
    plt.title(f"Visualización del Espacio Latente ({embeddings_file.stem})")
    plt.grid(True, alpha=0.2)
    
    out_file = FIGURES_DIR / output_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    
    logger.success(f"✨ Visualización latente guardada en: {out_file}")

if __name__ == "__main__":
    app()
