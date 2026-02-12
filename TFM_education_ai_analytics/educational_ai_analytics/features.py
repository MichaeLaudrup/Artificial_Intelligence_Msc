import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import typer

from educational_ai_analytics.config import PROCESSED_DATA_DIR

class FeatureEngineer:
    def __init__(self):
        # Almacén de escaladores entrenados (Fit on Train)
        self.scalers = {}
        self.medians = {}
        self.encoders = {}
        
        # Mapa ordinal fijo para IMD Band
        self.imd_map = {
            '0-10%': 0, '10-20%': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
            '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9,
            'Unknown': -1
        }
        
        # Mapa ordinal para Nivel Educativo
        self.education_map = {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        }
        
        # Mapa ordinal para Edad
        self.age_map = {
            '0-35': 0, 
            '35-55': 1, 
            '55<=': 2
        }
        
        # Mapas binarios
        self.gender_map = {'M': 0, 'F': 1}
        self.disability_map = {'N': 0, 'Y': 1}
        
    def _create_week_buckets(self, df_source, value_col, agg_func='sum', fill_val=0):
        """Genera matriz semanal dinamica (-2 a 35)."""
        # Pivotar
        if agg_func == 'sum':
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].sum().unstack(fill_value=fill_val)
        else:
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].mean().unstack(fill_value=fill_val)
            
        # Bucketing Logic
        ts_buckets = pd.DataFrame(index=ts_pivot.index)
        available_weeks = sorted(ts_pivot.columns.tolist())
        
        WEEK_START, WEEK_END = -2, 35
        cols_prev = [c for c in available_weeks if c < WEEK_START]
        cols_post = [c for c in available_weeks if c > WEEK_END]
        
        # Pre-Course
        if cols_prev:
            if agg_func == 'sum':
                ts_buckets[f'{value_col}_w_prev'] = ts_pivot[cols_prev].sum(axis=1)
            else:
                ts_buckets[f'{value_col}_w_prev'] = ts_pivot[cols_prev].mean(axis=1)
        else:
            ts_buckets[f'{value_col}_w_prev'] = fill_val

        # Core Weeks
        for c in range(WEEK_START, WEEK_END + 1):
            col_target = f'{value_col}_w_{c}' if c >= 0 else f'{value_col}_w_neg{abs(c)}'
            if c in ts_pivot.columns:
                ts_buckets[col_target] = ts_pivot[c]
            else:
                ts_buckets[col_target] = fill_val

        # Post-Course
        if cols_post:
            if agg_func == 'sum':
                ts_buckets[f'{value_col}_w_post'] = ts_pivot[cols_post].sum(axis=1)
            else:
                ts_buckets[f'{value_col}_w_post'] = ts_pivot[cols_post].mean(axis=1)
        else:
            ts_buckets[f'{value_col}_w_post'] = fill_val
            
        return ts_buckets

    def process_split(self, split_name: str, df_students, df_assessments, df_interactions, is_train=False):
        """Procesa un split completo (Train/Val/Test)."""
        logger.info(f"Ingeniería de Características para: {split_name} (Train={is_train})")
        
        # Generar ID único compuesto si no existe
        if 'unique_id' not in df_students.columns:
             df_students['unique_id'] = (df_students['id_student'].astype(str) + '_' + 
                                         df_students['code_module'] + '_' + 
                                         df_students['code_presentation'])

        # Asegurar unique_id en otras tablas
        for df in [df_assessments, df_interactions]:
            if 'unique_id' not in df.columns and 'id_student' in df.columns:
                df['unique_id'] = (df['id_student'].astype(str) + '_' + 
                                   df['code_module'] + '_' + 
                                   df['code_presentation'])
        
        # Crear columna 'week' a partir de 'date' (día relativo al inicio del curso)
        # week = date // 7 (Semana 0 = Días 0-6, Semana 1 = Días 7-13, etc.)
        if 'week' not in df_interactions.columns:
            df_interactions['date'] = pd.to_numeric(df_interactions['date'], errors='coerce')
            df_interactions['week'] = (df_interactions['date'] // 7).astype('Int64')
        
        if 'week' not in df_assessments.columns:
            df_assessments['date'] = pd.to_numeric(df_assessments['date'], errors='coerce')
            df_assessments['week'] = (df_assessments['date'] // 7).astype('Int64')
        
        all_ids = df_students['unique_id'].unique()
        
        # =========================================================
        # 1. SERIES TEMPORALES DE ACTIVIDAD (CLICKS)
        # =========================================================
        logger.info("- Generando TS Clicks...")
        ts_clicks = self._create_week_buckets(df_interactions, 'sum_click', agg_func='sum', fill_val=0)
        ts_clicks = ts_clicks.reindex(all_ids, fill_value=0)
        
        # Log1p Transformation (Normalización de distribución Power-Law)
        ts_clicks_log = np.log1p(ts_clicks)
        
        # Feature: Regularidad (Std Dev de series original)
        click_std = ts_clicks.std(axis=1)
        click_std.name = 'click_std'
        
        # =========================================================
        # 2. SERIES TEMPORALES DE RENDIMIENTO (SCORES + PROCRASTINATION)
        # =========================================================
        logger.info("- Generando TS Performance & Procrastination...")
        
        # 2A. Scores (Multicanal por tipo: TMA, CMA)
        # NOTA: Exam scores suelen ser al final, cuidado con leakage si es input. Mejor usar solo TMA/CMA continuo.
        df_assess_clean = df_assessments.dropna(subset=['score']) 
        df_assess_clean['score'] = pd.to_numeric(df_assess_clean['score'])
        
        ts_perf_list = []
        for atype in ['TMA', 'CMA']:
            df_type = df_assess_clean[df_assess_clean['assessment_type'] == atype]
            if not df_type.empty:
                ts = self._create_week_buckets(df_type, 'score', agg_func='mean', fill_val=0) # 0 = No presentado/Sin tarea
                ts.columns = [f'{c}_{atype}' for c in ts.columns] # Rename cols
                ts_perf_list.append(ts)
        
        if ts_perf_list:
            ts_performance = pd.concat(ts_perf_list, axis=1).reindex(all_ids, fill_value=0)
        else:
            ts_performance = pd.DataFrame(index=all_ids) # Empty

        # 2B. Procrastination (Days Early)
        df_assess_dates = df_assessments.dropna(subset=['date', 'date_submitted']).copy()
        df_assess_dates['date'] = pd.to_numeric(df_assess_dates['date'], errors='coerce')
        df_assess_dates['date_submitted'] = pd.to_numeric(df_assess_dates['date_submitted'], errors='coerce')
        df_assess_dates['days_early'] = df_assess_dates['date'] - df_assess_dates['date_submitted']
        
        ts_procrastination = self._create_week_buckets(df_assess_dates, 'days_early', agg_func='mean', fill_val=0)
        ts_procrastination = ts_procrastination.reindex(all_ids, fill_value=0)
        
        # =========================================================
        # 3. FEATURES ESTÁTICAS (CONTEXTO + DEMOGRÁFICOS)
        # =========================================================
        logger.info("- Generando Features Estáticas...")
        
        df_static = df_students.set_index('unique_id').copy()
        
        # Mapeos
        df_static['imd_band_numeric'] = df_static['imd_band'].map(self.imd_map).fillna(-1)
        
        # Créditos (Si existen en dataframe merged, sino 0)
        if 'credits' not in df_static.columns: df_static['credits'] = 0
        df_static['credits'] = pd.to_numeric(df_static['credits'], errors='coerce').fillna(0)
        
        # Duración
        df_static['module_presentation_length'] = pd.to_numeric(df_static['module_presentation_length'], errors='coerce').fillna(0)
        
        # Registration Delay
        df_static['date_registration'] = pd.to_numeric(df_static['date_registration'], errors='coerce').fillna(0)
        
        # --- DEMOGRÁFICOS ---
        df_static['education_level'] = df_static['highest_education'].map(self.education_map).fillna(1) # Default: Lower Than A Level
        df_static['age_numeric'] = df_static['age_band'].map(self.age_map).fillna(0)
        df_static['gender_bool'] = df_static['gender'].map(self.gender_map).fillna(0)
        df_static['disability_bool'] = df_static['disability'].map(self.disability_map).fillna(0)
        
        # Selección Final Estáticas
        static_cols = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 
            'education_level', 'age_numeric', 'gender_bool', 'disability_bool'
        ]
        X_static = df_static[static_cols].copy()
        
        # Añadir Regularidad (Click Std)
        X_static = X_static.join(click_std).fillna(0)
        
        # =========================================================
        # 4. NORMALIZACIÓN (FIT ON TRAIN, TRANSFORM ALL)
        # =========================================================
        
        # Seleccionamos solo las columnas numéricas para escalar
        cols_to_scale = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 'click_std',
            'education_level', 'age_numeric' # Gender y Disability son 0/1, no hace falta escalar pero no daña.
        ]
        # Aseguramos que existan
        cols_final = [c for c in cols_to_scale if c in X_static.columns]
        
        if is_train:
            logger.info("  * Entrenando Escaladores (MinMaxScaler 0-1)...")
            scaler = MinMaxScaler()
            # Fit solo en Train
            X_scaled_array = scaler.fit_transform(X_static[cols_final])
            self.scalers['static'] = scaler
            
            # Reasignamos valores escalados
            X_static[cols_final] = X_scaled_array
            
        else:
            logger.info("  * Aplicando Escaladores (Transform)...")
            if 'static' in self.scalers:
                scaler = self.scalers['static']
                # Transform en Val/Test usando rangos de Train
                X_scaled_array = scaler.transform(X_static[cols_final])
                X_static[cols_final] = X_scaled_array
            else:
                logger.warning("No hay escalador entrenado. Usando datos crudos en Test/Val.")
        
        # Target Mapping (Multiclase 4 Clases)
        # 0 = Pass (Aprobado)
        # 1 = Distinction (Sobresaliente)
        # 2 = Fail (Suspenso) -> Requiere Refuerzo Académico
        # 3 = Withdrawn (Abandono) -> Requiere Retención/Motivación
        target_map = {
            'Pass': 0, 
            'Distinction': 1, 
            'Fail': 2, 
            'Withdrawn': 3
        }
        
        return {
            'ts_clicks': ts_clicks_log,
            'ts_performance': ts_performance,
            'ts_procrastination': ts_procrastination,
            'static_features': X_static, # Ahora normalizado 0-1
            'target': df_students.set_index('unique_id')['final_result'].map(target_map)
        }

app = typer.Typer()


def run_feature_pipeline(input_dir: Path = PROCESSED_DATA_DIR):
    """Ejecuta el pipeline completo leyendo de data/processed/"""
    
    if not input_dir.exists():
        logger.error(f"No se encuentra {input_dir}. Ejecuta dataset.py primero.")
        raise FileNotFoundError(f"Directorio no encontrado: {input_dir}")

    engineer = FeatureEngineer()
    
    # Orden estricto: Primero Train para entrenar escaladores
    splits = ['training', 'validation', 'test']
    
    for split in splits:
        split_dir = input_dir / split
        if not split_dir.exists(): 
            logger.warning(f"Split '{split}' no encontrado en {input_dir}. Saltando.")
            continue
        
        # Cargar raw data del split
        df_stud = pd.read_csv(split_dir / "students.csv")
        df_assess = pd.read_csv(split_dir / "assessments.csv")
        df_inter = pd.read_csv(split_dir / "interactions.csv")
        
        # Procesar
        is_train_flag = (split == 'training')
        features_dict = engineer.process_split(split, df_stud, df_assess, df_inter, is_train=is_train_flag)
        
        # Guardar Features Generadas
        save_dir = split_dir / "features"
        if save_dir.exists():
            import shutil
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Guardando features en {save_dir}...")
        for name, df in features_dict.items():
            df.to_csv(save_dir / f"{name}.csv")
            
    logger.success("Pipeline de Ingeniería de Características completado con éxito.")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
):
    """
    Orquesta el pipeline de Feature Engineering.
    Lee los splits de data/processed/ y genera features en data/processed/{split}/features/
    """
    logger.info("Iniciando pipeline de Feature Engineering...")
    run_feature_pipeline(input_path)
    logger.success("Pipeline de Feature Engineering finalizado correctamente.")


if __name__ == "__main__":
    app()
