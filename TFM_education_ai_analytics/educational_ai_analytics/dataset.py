from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import requests
import zipfile
import shutil

from educational_ai_analytics.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, OULAD_DATASET_URL

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "OULAD_dataset" / "oulad_dataset.zip",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # Limpieza previa de la carpeta para evitar conflictos
    if input_path.parent.exists():
        logger.info(f"Limpiando carpeta antigua de OULAD")
        shutil.rmtree(input_path.parent)

    input_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(OULAD_DATASET_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(input_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(input_path.parent)


if __name__ == "__main__":
    app()
