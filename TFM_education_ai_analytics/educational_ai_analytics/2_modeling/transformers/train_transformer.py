import typer
from loguru import logger

app = typer.Typer()

@app.command()
def main():
    """
    Entrena el modelo Transformer (Fase futura del TFM).
    """
    logger.info("Iniciando entrenamiento del Transformer...")
    # TODO: Implementar arquitectura Transformer
    logger.warning("Fase de Transformer no implementada todavía.")

if __name__ == "__main__":
    app()
