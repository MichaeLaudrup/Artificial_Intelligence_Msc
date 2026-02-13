import pandas as pd
from loguru import logger
import typer
from educational_ai_analytics.config import EMBEDDINGS_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    method: str = "kmeans",
    use_ae: bool = True
):
    """
    Entrena algoritmos de clustering sobre los embeddings generados.
    """
    file_name = "latent_ae.csv" if use_ae else "latent_pca.csv"
    input_path = EMBEDDINGS_DATA_DIR / "training" / file_name
    
    if not input_path.exists():
        logger.error(f"No se encuentran embeddings en {input_path}. Ejecuta 'make encode' primero.")
        return

    logger.info(f"Cargando embeddings desde {file_name}...")
    df = pd.read_csv(input_path, index_col=0)
    
    logger.info(f"Iniciando clustering con el método: {method}...")
    # TODO: Implementar lógica de K-Means / DBSCAN aquí
    
    logger.success("Clustering finalizado (Template).")

if __name__ == "__main__":
    app()
