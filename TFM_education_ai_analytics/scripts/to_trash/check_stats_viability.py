import pandas as pd
import numpy as np
import os

# Configuración de rutas
DATA_DIR = "data/2_processed"
SETS = ["training", "validation", "test"]

def check_set(set_name):
    path = os.path.join(DATA_DIR, set_name, "interactions.csv")
    if not os.path.exists(path):
        print(f"⚠️  No se encuentra: {path}")
        return
    
    print(f"\n--- Analizando Set: {set_name.upper()} ---")
    df = pd.read_csv(path)
    
    # Filtrar pre-start como en el builder original (date < 0)
    pre = df[df["date"] < 0].copy()
    
    if pre.empty:
        print("❌ El conjunto no tiene actividad pre-start (date < 0)")
        return

    # Crear unique_id para agrupar por estudiante y curso
    pre["course_key"] = pre["code_module"] + "_" + pre["code_presentation"]
    pre["unique_id"] = pre["id_student"].astype(str) + "_" + pre["course_key"]

    # Calcular sumas por estudiante
    student_stats = pre.groupby("unique_id")["sum_click"].sum()
    
    # Calcular estadísticas por CURSO (que es lo que usa _apply_course_norm)
    course_stats = pre.groupby("course_key")["sum_click"].agg(["mean", "std", "count"])
    
    # Ver si hay cursos con std=0 o NaN (esto es lo que queremos ahorrarnos)
    broken_courses = course_stats[(course_stats["std"] == 0) | (course_stats["std"].isna())]
    
    print(f"Total Cursos: {len(course_stats)}")
    print(f"Cursos 'Rotos' (Sin desviación estándar): {len(broken_courses)}")
    
    if len(broken_courses) > 0:
        print("Ejemplos de cursos rotos (pocos alumnos o clicks idénticos):")
        print(broken_courses.head())
    else:
        print("✅ Todos los cursos tienen media y desviación válida.")

if __name__ == "__main__":
    for s in SETS:
        check_set(s)
