from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import requests
import zipfile
import shutil

import pandas as pd
from educational_ai_analytics.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, OULAD_DATASET_URL

app = typer.Typer()


def process_data(input_dir: Path, output_file: Path):
    """
    Carga los datos crudos (CSV), realiza uniones básicas y limpieza, 
    y guarda el dataset procesado.
    """
    logger.info(f"Procesando datos desde {input_dir}...")
    
    student_info_path = input_dir / "studentInfo.csv"
    courses_path = input_dir / "courses.csv"
    registration_path = input_dir / "studentRegistration.csv"

    if not all(p.exists() for p in [student_info_path, courses_path, registration_path]):
        logger.error("No se encontraron los archivos CSV necesarios (studentInfo, courses, studentRegistration).")
        raise FileNotFoundError("Faltan archivos CSV del dataset OULAD.")

    logger.info("Cargando CSVs...")
    
    # --- GRUPO 1: Demográficos y Registro ---
    df_student = pd.read_csv(input_dir / "studentInfo.csv")
    df_courses = pd.read_csv(input_dir / "courses.csv")
    df_registration = pd.read_csv(input_dir / "studentRegistration.csv")

    # --- GRUPO 2: Evaluaciones ---
    df_assessments = pd.read_csv(input_dir / "assessments.csv")
    df_student_assessments = pd.read_csv(input_dir / "studentAssessment.csv")

    # --- GRUPO 3: Interacciones (VLE) ---
    df_vle = pd.read_csv(input_dir / "vle.csv")
    df_student_vle = pd.read_csv(input_dir / "studentVle.csv")
    
    logger.info("Procesando datos de Estudiantes...")

    df_students_merged = pd.merge(
        df_student, 
        df_registration, 
        on=['code_module', 'code_presentation', 'id_student'], 
        how='left'
    )

    df_students_merged = pd.merge(
        df_students_merged,
        df_courses,
        on=['code_module', 'code_presentation'],
        how='left'
    )

    df_students_merged.replace('?', pd.NA, inplace=True)
    
    # ---------------------------------------------------------
    # PROCESAMIENTO 2: Evaluaciones (Enriquecer con metadatos)
    # ---------------------------------------------------------
    logger.info("Procesando datos de Evaluaciones...")
    df_assessments_merged = pd.merge(
        df_student_assessments,
        df_assessments,
        on=['id_assessment'],
        how='left'
    )
    df_assessments_merged.replace('?', pd.NA, inplace=True)

    logger.info("Procesando datos de Interacciones (VLE)...")
    df_interactions_merged = pd.merge(
        df_student_vle,
        df_vle,
        on=['id_site', 'code_module', 'code_presentation'],
        how='left'
    )
    df_interactions_merged.replace('?', pd.NA, inplace=True)

    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Guardando datasets procesados en {output_dir}...")
    
    df_students_merged.to_csv(output_dir / "students_processed.csv", index=False)
    df_assessments_merged.to_csv(output_dir / "assessments_processed.csv", index=False)
    df_interactions_merged.to_csv(output_dir / "interactions_processed.csv", index=False)
    
    logger.success("Procesamiento completado. Archivos generados:")
    logger.success(f" - students_processed.csv ({df_students_merged.shape})")
    logger.success(f" - assessments_processed.csv ({df_assessments_merged.shape})")
    logger.success(f" - interactions_processed.csv ({df_interactions_merged.shape})")


def download_dataset(url: str, dest_path: Path, force: bool = True):
    """
    Downloads the dataset from the URL to the destination path.
    If force is True, clears the parent directory first.
    """
    if dest_path.parent.exists() and force:
        logger.info(f"Limpiando carpeta antigua de OULAD: {dest_path.parent}")
        shutil.rmtree(dest_path.parent)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        logger.info(f"El archivo {dest_path} ya existe. Saltando descarga.")
        return

    logger.info(f"Descargando datos desde {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Descargando") as progress_bar:
            with open(dest_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                    
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Descarga incompleta o fallida.")
            
        logger.success("Descarga completada.")
        
    except Exception as e:
        logger.error(f"Error al descargar el dataset: {e}")
        raise


def extract_dataset(zip_path: Path, extract_to: Path):
    """
    Extracts the zip file to the specified directory.
    """
    logger.info(f"Extrayendo {zip_path} en {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.success("Extracción completada.")
    except zipfile.BadZipFile:
        logger.error(f"El archivo {zip_path} no es un zip válido.")
        raise


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "OULAD_dataset" / "oulad_dataset.zip",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Orquesta el pipeline del dataset: Descarga, Extracción, Procesamiento
    """
    logger.info("Iniciando pipeline de datos...")

    # 1. Descarga
    download_dataset(OULAD_DATASET_URL, input_path)

    # 2. Extracción
    extract_dataset(input_path, input_path.parent)

    # 3. Procesamiento
    process_data(input_path.parent, output_path)
    
    logger.success("Pipeline de datos finalizado correctamente.")


if __name__ == "__main__":
    app()
