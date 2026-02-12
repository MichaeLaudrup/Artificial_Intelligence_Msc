from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import requests
import zipfile
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from educational_ai_analytics.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, OULAD_DATASET_URL

app = typer.Typer()


def split_data_by_student(df_students, df_assessments, df_interactions, output_dir: Path):
    """
    Divide los datos en train, validation y test basándose en IDs de estudiantes únicos
    para evitar data leakage.
    Distribución: 70% Train, 15% Validation, 15% Test
    """
    logger.info("Iniciando división de datos (Splitting)...")
    
    # Obtener IDs únicos de estudiantes
    unique_students = df_students['id_student'].unique()
    
    # Primera división: Train (70%) vs Temp (30%)
    train_ids, temp_ids = train_test_split(unique_students, test_size=0.3, random_state=42)
    
    # Segunda división: Val (15%) vs Test (15%) - (El 50% del 30% restante es 15% del total)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    logger.info(f"Estudiantes únicos: {len(unique_students)}")
    logger.info(f" - Train: {len(train_ids)} ({len(train_ids)/len(unique_students):.1%})")
    logger.info(f" - Validation: {len(val_ids)} ({len(val_ids)/len(unique_students):.1%})")
    logger.info(f" - Test: {len(test_ids)} ({len(test_ids)/len(unique_students):.1%})")

    splits = {
        "training": train_ids,
        "validation": val_ids,
        "test": test_ids
    }

    for split_name, ids in splits.items():
        # Guardar en subcarpetas dentro de 'data/processed/'
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando conjunto '{split_name}' en {split_dir}...")
        
        # Filtrar tablas por ID de estudiante
        df_stud_split = df_students[df_students['id_student'].isin(ids)]
        df_assess_split = df_assessments[df_assessments['id_student'].isin(ids)]
        df_inter_split = df_interactions[df_interactions['id_student'].isin(ids)]
        
        # Guardar CSVs
        df_stud_split.to_csv(split_dir / "students.csv", index=False)
        df_assess_split.to_csv(split_dir / "assessments.csv", index=False)
        df_inter_split.to_csv(split_dir / "interactions.csv", index=False)


def save_interim_data(df_students, df_assessments, df_interactions):
    """
    Guarda los datos mergeados pero SIN limpiar en data/interim/.
    Estos son los datos que usa el notebook de EDA para sacar conclusiones.
    """
    interim_dir = INTERIM_DATA_DIR
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Guardando datos intermedios (para EDA) en {interim_dir}...")
    
    df_students.to_csv(interim_dir / "students.csv", index=False)
    df_assessments.to_csv(interim_dir / "assessments.csv", index=False)
    df_interactions.to_csv(interim_dir / "interactions.csv", index=False)
    
    logger.success("Datos intermedios guardados.")


def apply_eda_cleaning(df_students, df_assessments, df_interactions):
    """
    Aplica las reglas de limpieza derivadas del Análisis Exploratorio de Datos (EDA).
    Recibe los DataFrames mergeados y devuelve los DataFrames limpios.
    """
    logger.info("Aplicando limpieza basada en conclusiones del EDA...")

    # --- ESTUDIANTES ---
    # 1. Imputar Nulos en Fecha de Registro (MEDIANA por presentación)
    df_students['date_registration'] = pd.to_numeric(df_students['date_registration'], errors='coerce')
    median_registration = df_students.groupby('code_presentation')['date_registration'].transform('median')
    df_students['date_registration'] = df_students['date_registration'].fillna(median_registration)
    
    # 2. Imputar nulos en IMD Band como 'Unknown'
    df_students['imd_band'] = df_students['imd_band'].fillna('Unknown')

    # 3. Conversión de tipos numéricos
    for col in ['date_unregistration', 'module_presentation_length']:
        if col in df_students.columns:
            df_students[col] = pd.to_numeric(df_students[col], errors='coerce')

    # --- EVALUACIONES ---
    # Eliminar filas sin nota
    df_assessments.dropna(subset=['score'], inplace=True)
    df_assessments['score'] = pd.to_numeric(df_assessments['score'])

    logger.success("Limpieza EDA completada.")
    return df_students, df_assessments, df_interactions


def process_data(input_dir: Path, output_file: Path):
    """
    Pipeline completo:
    1. Carga los CSVs crudos
    2. Realiza merges entre tablas
    3. Guarda versión intermedia (interim) para EDA
    4. Aplica limpieza basada en EDA
    5. Divide en train/validation/test y guarda en processed
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
    
    # =========================================================
    # FASE 1: MERGES (unir tablas relacionales)
    # =========================================================
    logger.info("Procesando datos de Estudiantes...")
    df_students_merged = pd.merge(
        df_student, df_registration, 
        on=['code_module', 'code_presentation', 'id_student'], how='left'
    )
    df_students_merged = pd.merge(
        df_students_merged, df_courses,
        on=['code_module', 'code_presentation'], how='left'
    )
    df_students_merged.replace('?', pd.NA, inplace=True)

    logger.info("Procesando datos de Evaluaciones...")
    df_assessments_merged = pd.merge(
        df_student_assessments, df_assessments,
        on=['id_assessment'], how='left'
    )
    df_assessments_merged.replace('?', pd.NA, inplace=True)

    logger.info("Procesando datos de Interacciones (VLE)...")
    df_interactions_merged = pd.merge(
        df_student_vle, df_vle,
        on=['id_site', 'code_module', 'code_presentation'], how='left'
    )
    df_interactions_merged.replace('?', pd.NA, inplace=True)

    # =========================================================
    # FASE 2: GUARDAR DATOS INTERMEDIOS (para EDA, sin limpiar)
    # =========================================================
    save_interim_data(df_students_merged, df_assessments_merged, df_interactions_merged)

    # =========================================================
    # FASE 3: APLICAR LIMPIEZA BASADA EN EDA
    # =========================================================
    df_students_merged, df_assessments_merged, df_interactions_merged = apply_eda_cleaning(
        df_students_merged, df_assessments_merged, df_interactions_merged
    )

    # =========================================================
    # FASE 4: DIVIDIR Y GUARDAR EN PROCESSED
    # =========================================================
    output_dir = output_file.parent
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Carpeta {output_dir} limpiada.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_data_by_student(df_students_merged, df_assessments_merged, df_interactions_merged, output_dir)
    
    logger.success("Pipeline completado. Datos intermedios, procesados y divididos.")


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
