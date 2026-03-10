import os
import warnings

# Silenciar warnings de Protobuf y logs de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import requests
import zipfile
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from educational_ai_analytics.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    OULAD_DATASET_URL,
    WITH_SYNTHETIC,
)

app = typer.Typer()

def clean_data_directory():
    """Borra todo el contenido del directorio DATA_DIR para empezar de cero."""
    if DATA_DIR.exists():
        logger.info(f"🧹 Limpiando todo el contenido de {DATA_DIR}...")
        for item in DATA_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.success("Directorio de datos limpio.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_data_splits(df_students, df_assessments, df_interactions):
    """
    Divide los datos en train, validation y test basándose en IDs de estudiantes únicos
    para evitar data leakage.
    Distribución: 70% Train, 15% Validation, 15% Test
    
    Retorna: diccionario con los splits {'training': (dfs...), 'validation': ..., 'test': ...}
    """
    logger.info("Iniciando división de datos (Splitting)...")
    
    # Obtener IDs únicos de estudiantes y sus etiquetas para estratificación
    # Usamos el primer 'final_result' del estudiante como proxy para estratificar por clase
    student_labels = df_students.groupby('id_student')['final_result'].first()
    unique_students = student_labels.index
    labels = student_labels.values
    
    # Primera división: Train (70%) vs Temp (30%) - ESTRATIFICADO
    train_ids, temp_ids, _, temp_labels = train_test_split(
        unique_students, 
        labels,
        test_size=0.3, 
        random_state=42,
        stratify=labels
    )
    
    # Segunda división: Val (15%) vs Test (15%) - ESTRATIFICADO
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_labels
    )
    
    logger.info(f"Estudiantes únicos: {len(unique_students)}")
    logger.info(f" - Train: {len(train_ids)} ({len(train_ids)/len(unique_students):.1%})")
    logger.info(f" - Validation: {len(val_ids)} ({len(val_ids)/len(unique_students):.1%})")
    logger.info(f" - Test: {len(test_ids)} ({len(test_ids)/len(unique_students):.1%})")

    ids_map = {
        "training": train_ids,
        "validation": val_ids,
        "test": test_ids
    }
    
    splits = {}
    for split_name, ids in ids_map.items():
        splits[split_name] = {
            'students': df_students[df_students['id_student'].isin(ids)].copy(),
            'assessments': df_assessments[df_assessments['id_student'].isin(ids)].copy(),
            'interactions': df_interactions[df_interactions['id_student'].isin(ids)].copy()
        }
    
    return splits


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


def apply_structural_cleaning(df_students, df_assessments, df_interactions):
    """
    Aplica limpieza ESTRUCTURAL:
    - Conversión de tipos.
    - Manejo de categorías (Unknown).
    - Eliminación de filas inválidas (sin nota).
    NO realiza imputaciones estadísticas que dependan de la distribución global.
    """
    logger.info("Aplicando limpieza ESTRUCTURAL...")

    # --- ESTUDIANTES ---
    # 1. Conversión de tipos
    # date_registration se convierte a numérico, pero NO se imputa aquí todavía (data leakage)
    df_students['date_registration'] = pd.to_numeric(df_students['date_registration'], errors='coerce')
    
    for col in ['date_unregistration', 'module_presentation_length']:
        if col in df_students.columns:
            df_students[col] = pd.to_numeric(df_students[col], errors='coerce')

    # 2. Imputar nulos en IMD Band como 'Unknown' (Categórico constante)
    df_students['imd_band'] = df_students['imd_band'].fillna('Unknown')

    # --- EVALUACIONES ---
    # Eliminar filas sin nota (Estructural: datos corruptos/incompletos inservibles)
    df_assessments.dropna(subset=['score'], inplace=True)
    df_assessments['score'] = pd.to_numeric(df_assessments['score'])

    logger.success("Limpieza Estructural completada.")
    return df_students, df_assessments, df_interactions


def impute_statistical_data(df_students, registration_medians=None):
    """
    Realiza la imputación estadística de 'date_registration'.
    Si se pasan 'registration_medians', se usan (Validacion/Test).
    Si no (None), se calculan (Train) y se devuelven.
    """
    if registration_medians is None:
        # Modo FIT: Calcular medianas (Solo Train)
        logger.info("Calculando estadísticas de imputación (Fit en Train)...")
        registration_medians = df_students.groupby('code_presentation')['date_registration'].median()
    
    # Modo TRANSFORM: Aplicar imputación
    # Mapear la mediana correspondiente a cada presentación
    median_values = df_students['code_presentation'].map(registration_medians)
    
    # Fallback: Si hay alguna presentación en Val/Test que no estaba en Train (raro), 
    # usar la mediana global de las medianas de Train.
    if median_values.isna().any():
        global_median = registration_medians.median()
        median_values = median_values.fillna(global_median)
    
    df_students['date_registration'] = df_students['date_registration'].fillna(median_values)
    
    return df_students, registration_medians


def augment_training_withdrawn(split_payload):
    """
    Duplica estudiantes Withdrawn SOLO en training hasta igualar la clase mayoritaria.
    No es síntesis generativa real; es oversampling bootstrap con nuevos id_student.
    """
    students = split_payload['students']
    assessments = split_payload['assessments']
    interactions = split_payload['interactions']

    student_labels = students.groupby('id_student')['final_result'].first()
    class_counts = student_labels.value_counts()
    withdrawn_count = int(class_counts.get('Withdrawn', 0))
    target_count = int(class_counts.max()) if not class_counts.empty else 0

    if withdrawn_count == 0 or target_count <= withdrawn_count:
        logger.info(
            "WITH_SYNTHETIC activado, pero no hace falta oversampling de Withdrawn (current={} target={}).",
            withdrawn_count,
            target_count,
        )
        return split_payload

    needed = target_count - withdrawn_count
    withdrawn_ids = student_labels[student_labels == 'Withdrawn'].index.to_numpy()
    sampled_ids = pd.Series(withdrawn_ids).sample(n=needed, replace=True, random_state=42).tolist()

    numeric_ids = pd.to_numeric(students['id_student'], errors='coerce')
    next_id = int(numeric_ids.max()) + 1 if numeric_ids.notna().any() else int(len(student_labels) + 1)

    synthetic_students = []
    synthetic_assessments = []
    synthetic_interactions = []

    for offset, source_id in enumerate(sampled_ids):
        synthetic_id = next_id + offset

        student_rows = students[students['id_student'] == source_id].copy()
        assessment_rows = assessments[assessments['id_student'] == source_id].copy()
        interaction_rows = interactions[interactions['id_student'] == source_id].copy()

        student_rows['id_student'] = synthetic_id
        assessment_rows['id_student'] = synthetic_id
        interaction_rows['id_student'] = synthetic_id

        synthetic_students.append(student_rows)
        synthetic_assessments.append(assessment_rows)
        synthetic_interactions.append(interaction_rows)

    split_payload['students'] = pd.concat([students] + synthetic_students, ignore_index=True)
    split_payload['assessments'] = pd.concat([assessments] + synthetic_assessments, ignore_index=True)
    split_payload['interactions'] = pd.concat([interactions] + synthetic_interactions, ignore_index=True)

    logger.info(
        "🧪 WITH_SYNTHETIC=True -> añadidos {} estudiantes Withdrawn al training ({} -> {}).",
        needed,
        withdrawn_count,
        target_count,
    )
    return split_payload


def process_data(input_dir: Path, output_file: Path):
    """
    Pipeline completo:
    1. Carga los CSVs crudos
    2. Realiza merges entre tablas
    3. Guarda versión intermedia (interim) para EDA
    4. Aplica limpieza ESTRUCTURAL (pre-split)
    5. Divide en train/validation/test
    6. Aplica limpieza ESTADÍSTICA (imputación) usando stats de TRAIN
    7. Guarda en processed
    """
    logger.info(f"Procesando datos desde {input_dir}...")
    
    student_info_path = input_dir / "studentInfo.csv"
    courses_path = input_dir / "courses.csv"
    registration_path = input_dir / "studentRegistration.csv"

    if not all(p.exists() for p in [student_info_path, courses_path, registration_path]):
        logger.error("No se encontraron los archivos CSV necesarios.")
        raise FileNotFoundError("Faltan archivos CSV del dataset OULAD.")

    logger.info("Cargando CSVs...")
    
    # --- Carga ---
    df_student = pd.read_csv(input_dir / "studentInfo.csv")
    df_courses = pd.read_csv(input_dir / "courses.csv")
    df_registration = pd.read_csv(input_dir / "studentRegistration.csv")
    df_assessments = pd.read_csv(input_dir / "assessments.csv")
    df_student_assessments = pd.read_csv(input_dir / "studentAssessment.csv")
    df_vle = pd.read_csv(input_dir / "vle.csv")
    df_student_vle = pd.read_csv(input_dir / "studentVle.csv")
    
    # =========================================================
    # FASE 1: MERGES
    # =========================================================
    logger.info("Realizando Merges...")
    df_students_merged = pd.merge(
        df_student, df_registration, 
        on=['code_module', 'code_presentation', 'id_student'], how='left'
    )
    df_students_merged = pd.merge(
        df_students_merged, df_courses,
        on=['code_module', 'code_presentation'], how='left'
    )
    df_students_merged.replace('?', pd.NA, inplace=True)

    df_assessments_merged = pd.merge(
        df_student_assessments, df_assessments,
        on=['id_assessment'], how='left'
    )
    df_assessments_merged.replace('?', pd.NA, inplace=True)

    df_interactions_merged = pd.merge(
        df_student_vle, df_vle,
        on=['id_site', 'code_module', 'code_presentation'], how='left'
    )
    df_interactions_merged.replace('?', pd.NA, inplace=True)

    # =========================================================
    # FASE 2: INTERIM DATA
    # =========================================================
    save_interim_data(df_students_merged, df_assessments_merged, df_interactions_merged)

    # =========================================================
    # FASE 3: LIMPIEZA ESTRUCTURAL (Global)
    # =========================================================
    df_students_merged, df_assessments_merged, df_interactions_merged = apply_structural_cleaning(
        df_students_merged, df_assessments_merged, df_interactions_merged
    )

    # =========================================================
    # FASE 4: DIVISIÓN (SPLIT)
    # =========================================================
    splits = get_data_splits(df_students_merged, df_assessments_merged, df_interactions_merged)
    
    # =========================================================
    # FASE 5: LIMPIEZA ESTADÍSTICA (Fit on Train, Transform All)
    # =========================================================
    logger.info("Aplicando imputación estadística (evitando Data Leakage)...")
    
    # 1. Fit en Train
    train_students = splits['training']['students']
    train_students, registration_stats = impute_statistical_data(train_students, registration_medians=None)
    splits['training']['students'] = train_students # Update cleaned version
    
    logger.info("Estadísticas aprendidas en Train. Aplicando a Val/Test...")
    
    # 2. Transform Val & Test
    for split_name in ['validation', 'test']:
        df_stud = splits[split_name]['students']
        df_stud, _ = impute_statistical_data(df_stud, registration_medians=registration_stats)
        splits[split_name]['students'] = df_stud

    if WITH_SYNTHETIC:
        splits['training'] = augment_training_withdrawn(splits['training'])

    # =========================================================
    # FASE 6: GUARDADO FINAL
    # =========================================================
    root_output_dir = output_file.parent
    if root_output_dir.exists():
        shutil.rmtree(root_output_dir)
        logger.info(f"Limpiando directorio destino: {root_output_dir}")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        split_dir = root_output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando {split_name} en {split_dir}...")
        data['students'].to_csv(split_dir / "students.csv", index=False)
        data['assessments'].to_csv(split_dir / "assessments.csv", index=False)
        data['interactions'].to_csv(split_dir / "interactions.csv", index=False)
    
    logger.success(f"Pipeline completado sin Data Leakage. Datos guardados en {root_output_dir}")


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

    # 0. Limpieza total para empezar de cero
    clean_data_directory()

    # 1. Descarga
    download_dataset(OULAD_DATASET_URL, input_path)

    # 2. Extracción
    extract_dataset(input_path, input_path.parent)

    # 3. Procesamiento
    process_data(input_path.parent, output_path)
    
    logger.success("Pipeline de datos finalizado correctamente.")


if __name__ == "__main__":
    app()
