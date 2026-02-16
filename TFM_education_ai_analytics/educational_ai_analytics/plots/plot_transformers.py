import typer
from loguru import logger
from .style import set_style

app = typer.Typer(help="Visualizaciones para Transformers.")
set_style()

@app.command()
def placeholder():
    """Placeholder para futuras visualizaciones de Transformers."""
    logger.info("Visualizaciones de Transformers no implementadas todavía.")

if __name__ == "__main__":
    app()
