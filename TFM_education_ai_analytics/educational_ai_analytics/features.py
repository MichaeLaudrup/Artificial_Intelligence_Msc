<<<<<<< HEAD
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import typer

from educational_ai_analytics.config import PROCESSED_DATA_DIR


# =========================================================================
# CONSTANTES DE DISEÑO
# =========================================================================
WEEK_START, WEEK_END = -2, 35  # Rango de semanas lectivas (40 semanas)

# Mapa de canales de actividad (basado en hallazgos del EDA)
# Fuente: docs/conclusiones_eda.md - "Calidad vs Cantidad (Perfil de Navegación)"
ACTIVITY_MAP = {
    'content': ['oucontent', 'resource', 'url', 'page', 'subpage'],  # Estudio pasivo
    'social': ['forumng', 'oucollaborate', 'ouwiki'],                # Aprendizaje social
    'quiz': ['quiz', 'questionnaire'],                               # Autoevaluación
    'other': []  # Placeholder: captura todo lo no mapeado (homepage, glossary, etc.)
}

# Tipos de evaluación a procesar (Exam excluido por data leakage)
ASSESSMENT_TYPES = ['TMA', 'CMA']


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

    # =====================================================================
    # HELPERS DE BUCKETING TEMPORAL
    # =====================================================================

    def _bucket_week_columns(self, ts_pivot, prefix, agg_func='sum', fill_val=0):
        """
        Aplica la lógica de bucketing semanal a un DataFrame ya pivotado.
        Agrupa semanas < WEEK_START en 'prev' y > WEEK_END en 'post'.
        
        Args:
            ts_pivot: DataFrame pivotado con semanas como columnas
            prefix: Prefijo para los nombres de columna (ej: 'content', 'TMA_avg')
            agg_func: 'sum' o 'mean' para agregar prev/post
            fill_val: Valor de relleno para semanas sin datos
        
        Returns:
            DataFrame con columnas: {prefix}_prev, {prefix}_w_neg2, ..., {prefix}_w_35, {prefix}_post
        """
        ts_buckets = pd.DataFrame(index=ts_pivot.index)
        available_weeks = sorted(ts_pivot.columns.tolist())
        
        cols_prev = [c for c in available_weeks if c < WEEK_START]
        cols_post = [c for c in available_weeks if c > WEEK_END]
        
        # Pre-Course
        if cols_prev:
            if fill_val == -1:
                # Para scores: ignorar -1 (no entregado) al agregar
                ts_buckets[f'{prefix}_w_prev'] = (
                    ts_pivot[cols_prev].replace(-1, np.nan).mean(axis=1).fillna(fill_val)
                )
            elif agg_func == 'sum':
                ts_buckets[f'{prefix}_w_prev'] = ts_pivot[cols_prev].sum(axis=1)
            else:
                ts_buckets[f'{prefix}_w_prev'] = ts_pivot[cols_prev].mean(axis=1)
        else:
            ts_buckets[f'{prefix}_w_prev'] = fill_val

        # Core Weeks (-2 a 35)
        for w in range(WEEK_START, WEEK_END + 1):
            col_name = f'{prefix}_w_{w}' if w >= 0 else f'{prefix}_w_neg{abs(w)}'
            if w in ts_pivot.columns:
                ts_buckets[col_name] = ts_pivot[w]
            else:
                ts_buckets[col_name] = fill_val

        # Post-Course
        if cols_post:
            if fill_val == -1:
                ts_buckets[f'{prefix}_w_post'] = (
                    ts_pivot[cols_post].replace(-1, np.nan).mean(axis=1).fillna(fill_val)
                )
            elif agg_func == 'sum':
                ts_buckets[f'{prefix}_w_post'] = ts_pivot[cols_post].sum(axis=1)
            else:
                ts_buckets[f'{prefix}_w_post'] = ts_pivot[cols_post].mean(axis=1)
        else:
            ts_buckets[f'{prefix}_w_post'] = fill_val
            
        return ts_buckets

    def _create_week_buckets(self, df_source, value_col, agg_func='sum', fill_val=0):
        """Genera matriz semanal dinámica (-2 a 35). Wrapper legacy."""
        if agg_func == 'sum':
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].sum().unstack(fill_value=fill_val)
        else:
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].mean().unstack(fill_value=fill_val)
        
        return self._bucket_week_columns(ts_pivot, value_col, agg_func, fill_val)

    # =====================================================================
    # GENERACIÓN DE CLICKS MULTICANAL
    # =====================================================================

    def _generate_multichannel_clicks(self, df_interactions, all_ids):
        """
        Genera matriz de clics desglosada por tipo de actividad.
        
        Canales: content, social, quiz, other
        Cada canal tiene 40 columnas (prev + weeks -2..35 + post)
        Total: 4 × 40 = 160 columnas
        
        Se aplica transformación log1p para normalizar la distribución power-law.
        """
        logger.info("- Generando Clics Multicanal (4 canales)...")
        channels_list = []
        
        # Obtener todos los activity_types mapeados para calcular 'other'
        all_mapped_types = sum(ACTIVITY_MAP.values(), [])
        
        for channel_name, target_types in ACTIVITY_MAP.items():
            logger.info(f"  · Canal: {channel_name} ({len(target_types)} tipos de actividad)")
            
            # Filtrar interacciones por tipo
            if target_types:
                df_sub = df_interactions[df_interactions['activity_type'].isin(target_types)]
            else:
                # Canal 'other': todo lo que NO está mapeado
                df_sub = df_interactions[~df_interactions['activity_type'].isin(all_mapped_types)]
            
            # Pivotar por semana
            if df_sub.empty:
                # Canal vacío: crear DataFrame con ceros
                ts_channel = pd.DataFrame(0, index=all_ids, 
                                          columns=[f'{channel_name}_w_prev'] +
                                                   [f'{channel_name}_w_{w}' if w >= 0 else f'{channel_name}_w_neg{abs(w)}' 
                                                    for w in range(WEEK_START, WEEK_END + 1)] +
                                                   [f'{channel_name}_w_post'])
            else:
                ts_pivot = df_sub.groupby(['unique_id', 'week'])['sum_click'].sum().unstack(fill_value=0)
                ts_channel = self._bucket_week_columns(ts_pivot, channel_name, agg_func='sum', fill_val=0)
            
            # Reindexar para alinear con todos los estudiantes
            ts_channel = ts_channel.reindex(all_ids, fill_value=0)
            channels_list.append(ts_channel)
        
        # Concatenar todos los canales
        ts_multichannel = pd.concat(channels_list, axis=1)
        
        # Log1p Transformation (normaliza distribución power-law de clics)
        ts_multichannel_log = np.log1p(ts_multichannel)
        
        logger.info(f"  → Matriz Clics Multicanal: {ts_multichannel_log.shape}")
        return ts_multichannel_log, ts_multichannel

    # =====================================================================
    # GENERACIÓN DE RENDIMIENTO DERIVADO (AVG, RATE, TREND)
    # =====================================================================

    def _get_performance_base_matrix(self, df_assessments, assess_type, all_ids):
        """
        Genera la matriz base de notas crudas para un tipo de evaluación.
        Usa -1 como sentinel para "no entregado/sin tarea en esa semana".
        """
        df_type = df_assessments[df_assessments['assessment_type'] == assess_type].copy()
        
        if df_type.empty:
            # No hay evaluaciones de este tipo
            cols = ([f'{assess_type}_w_prev'] +
                    [f'{assess_type}_w_{w}' if w >= 0 else f'{assess_type}_w_neg{abs(w)}'
                     for w in range(WEEK_START, WEEK_END + 1)] +
                    [f'{assess_type}_w_post'])
            return pd.DataFrame(-1, index=all_ids, columns=cols)
        
        # Pivotar: promedio de notas por semana (si hay varias entregas en la misma semana)
        ts_pivot = df_type.groupby(['unique_id', 'week'])['score'].mean().unstack(fill_value=-1)
        
        # Aplicar bucketing con -1 como fill_val
        ts_base = self._bucket_week_columns(ts_pivot, assess_type, agg_func='mean', fill_val=-1)
        ts_base = ts_base.reindex(all_ids, fill_value=-1)
        
        return ts_base

    def _generate_performance_features(self, df_assessments, all_ids):
        """
        Genera features derivadas de rendimiento para cada tipo de evaluación.
        
        Para cada tipo (TMA, CMA) se generan 3 métricas × 40 semanas:
          - avg:   Promedio acumulado (running average) normalizado 0-1
          - rate:  Tasa de entrega acumulada (submissions / total posible)
          - trend: Tendencia (nota actual - promedio histórico) normalizada
        
        Total: 2 tipos × 3 métricas × 40 semanas = 240 columnas
        
        Convención de valores:
          -1 = No hay evaluación de ese tipo en esa semana (sentinel)
          0  = Hubo evaluación pero no se entregó / nota 0
        """
        logger.info("- Generando Rendimiento Multicanal (avg, rate, trend × TMA, CMA)...")
        
        # Limpiar: solo registros con score y excluir Exam (data leakage)
        df_assess_clean = df_assessments.dropna(subset=['score']).copy()
        df_assess_clean['score'] = pd.to_numeric(df_assess_clean['score'], errors='coerce')
        df_assess_clean = df_assess_clean[df_assess_clean['assessment_type'].isin(ASSESSMENT_TYPES)]
        
        performance_matrices = []
        
        for assess_type in ASSESSMENT_TYPES:
            logger.info(f"  · Procesando canal de evaluación: {assess_type}...")
            
            # A. Matriz Base (Notas crudas con buckets, -1 = sin evaluación)
            ts_base = self._get_performance_base_matrix(df_assess_clean, assess_type, all_ids)
            ts_raw = ts_base.replace(-1, np.nan)  # NaN para cálculos
            cols = ts_base.columns
            
            # B. Feature 1: Promedio Acumulado (Running Average)
            # Expanding mean a lo largo del tiempo, normalizado 0-1
            # Forward fill de NaN con el promedio anterior
            feat_avg = ts_raw.T.expanding().mean().T.fillna(0) / 100.0
            feat_avg.columns = [c.replace(f'{assess_type}_', f'{assess_type}_avg_') for c in cols]
            
            # C. Feature 2: Tasa de Entrega (Submission Rate acumulada)
            # 1 si entregó, 0 si no
            is_submitted = (~ts_raw.isna()).astype(int)
            # Acumulado de entregas
            feat_rate = is_submitted.T.cumsum().T
            # Normalizar por el máximo de entregas vistas
            max_possible = feat_rate.max()
            feat_rate = feat_rate.div(max_possible.replace(0, 1), axis=1)
            feat_rate.columns = [c.replace(f'{assess_type}_', f'{assess_type}_rate_') for c in cols]
            
            # D. Feature 3: Tendencia (Score Trend)
            # Nota actual - Promedio histórico (0 si no entregó)
            # Permite detectar mejoras/empeoramientos progresivos
            trend_vals = (ts_raw.fillna(0).values - (feat_avg.values * 100)) / 100.0
            feat_trend = pd.DataFrame(
                trend_vals, 
                index=ts_base.index,
                columns=[c.replace(f'{assess_type}_', f'{assess_type}_trend_') for c in cols]
            )
            
            # Concatenar las 3 métricas de este canal
            performance_matrices.extend([feat_avg, feat_rate, feat_trend])
        
        # Consolidar todo
        ts_performance = pd.concat(performance_matrices, axis=1)
        
        logger.info(f"  → Matriz Rendimiento Multicanal: {ts_performance.shape}")
        return ts_performance

    # =====================================================================
    # GENERACIÓN DE PROCRASTINACIÓN
    # =====================================================================

    def _generate_procrastination(self, df_assessments, all_ids):
        """
        Genera serie temporal de procrastinación (days_early).
        
        days_early = date_deadline - date_submitted
          > 0: Entrega anticipada (bien)
          = 0: Entrega al límite
          < 0: Entrega tardía (riesgo)
          -999 (sentinel interno): Sin entrega → se reemplaza por 0
        """
        logger.info("- Generando Canal de Procrastinación (Days Early)...")
        
        df_assess_dates = df_assessments.dropna(subset=['date', 'date_submitted']).copy()
        df_assess_dates['date'] = pd.to_numeric(df_assess_dates['date'], errors='coerce')
        df_assess_dates['date_submitted'] = pd.to_numeric(df_assess_dates['date_submitted'], errors='coerce')
        df_assess_dates = df_assess_dates.dropna(subset=['date']).copy()
        df_assess_dates['days_early'] = df_assess_dates['date'] - df_assess_dates['date_submitted']
        
        # Pivotar con sentinel -999 para "sin entrega"
        ts_pivot = (df_assess_dates.groupby(['unique_id', 'week'])['days_early']
                    .mean().unstack(fill_value=-999))
        
        # Bucketing manual con reemplazo de sentinel
        ts_buckets = pd.DataFrame(index=ts_pivot.index)
        available_weeks = sorted(ts_pivot.columns.tolist())
        
        cols_prev = [c for c in available_weeks if c < WEEK_START]
        cols_core = [c for c in available_weeks if WEEK_START <= c <= WEEK_END]
        cols_post = [c for c in available_weeks if c > WEEK_END]
        
        # Pre-Course
        ts_buckets['days_early_w_prev'] = (
            ts_pivot[cols_prev].replace(-999, np.nan).mean(axis=1).fillna(0)
            if cols_prev else 0
        )
        
        # Core Weeks
        for w in range(WEEK_START, WEEK_END + 1):
            col_name = f'days_early_w_{w}' if w >= 0 else f'days_early_w_neg{abs(w)}'
            if w in ts_pivot.columns:
                # Reemplazar sentinel -999 por 0 (sin entrega = neutral)
                # Nota de Diseño: 0 puede significar "Entrega al límite" O "Sin Tarea"
                # Esto es aceptable porque el canal de Rendimiento (Scores) desambigua:
                # si Score != -1, entonces el 0 en procrastinación = "Entrega al límite"
                ts_buckets[col_name] = ts_pivot[w].replace(-999, 0)
            else:
                ts_buckets[col_name] = 0
        
        # Post-Course
        ts_buckets['days_early_w_post'] = (
            ts_pivot[cols_post].replace(-999, np.nan).mean(axis=1).fillna(0)
            if cols_post else 0
        )
        
        ts_procrastination = ts_buckets.reindex(all_ids, fill_value=0)
        logger.info(f"  → Matriz Procrastinación: {ts_procrastination.shape}")
        return ts_procrastination

    # =====================================================================
    # FEATURES ESTÁTICAS
    # =====================================================================

    def _generate_static_features(self, df_students, click_std, all_ids, is_train=False):
        """
        Genera features estáticas: demográficas, socioeconómicas, y regularidad.
        Aplica MinMaxScaler (fit on train, transform on val/test).
        """
        logger.info("- Generando Features Estáticas...")
        
        df_static = df_students.set_index('unique_id').copy()
        
        # --- Mapeos Ordinales ---
        df_static['imd_band_numeric'] = df_static['imd_band'].map(self.imd_map).fillna(-1)
        
        # Créditos
        if 'credits' not in df_static.columns:
            df_static['credits'] = 0
        df_static['credits'] = pd.to_numeric(df_static['credits'], errors='coerce').fillna(0)
        
        # Duración del módulo
        df_static['module_presentation_length'] = pd.to_numeric(
            df_static['module_presentation_length'], errors='coerce'
        ).fillna(0)
        
        # Fecha de registro (delay respecto al inicio del curso)
        df_static['date_registration'] = pd.to_numeric(
            df_static['date_registration'], errors='coerce'
        ).fillna(0)
        
        # --- Demográficos ---
        df_static['education_level'] = df_static['highest_education'].map(self.education_map).fillna(1)
        df_static['age_numeric'] = df_static['age_band'].map(self.age_map).fillna(0)
        df_static['gender_bool'] = df_static['gender'].map(self.gender_map).fillna(0)
        df_static['disability_bool'] = df_static['disability'].map(self.disability_map).fillna(0)
        
        # --- Selección de columnas ---
        static_cols = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 
            'education_level', 'age_numeric', 'gender_bool', 'disability_bool'
        ]
        X_static = df_static[static_cols].copy()
        
        # Añadir Regularidad (Click Std - varianza de clics totales)
        X_static = X_static.join(click_std).fillna(0)
        
        # --- Normalización (Fit on Train, Transform All) ---
        cols_to_scale = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 'click_std',
            'education_level', 'age_numeric'
        ]
        cols_final = [c for c in cols_to_scale if c in X_static.columns]
        
        if is_train:
            logger.info("  * Entrenando Escaladores (MinMaxScaler 0-1)...")
            scaler = MinMaxScaler()
            X_static[cols_final] = scaler.fit_transform(X_static[cols_final])
            self.scalers['static'] = scaler
        else:
            logger.info("  * Aplicando Escaladores (Transform)...")
            if 'static' in self.scalers:
                scaler = self.scalers['static']
                X_static[cols_final] = scaler.transform(X_static[cols_final])
            else:
                logger.warning("No hay escalador entrenado. Usando datos crudos en Test/Val.")
        
        logger.info(f"  → Features Estáticas: {X_static.shape}")
        return X_static

    # =====================================================================
    # ORQUESTADOR PRINCIPAL
    # =====================================================================

    def process_split(self, split_name: str, df_students, df_assessments, df_interactions, is_train=False):
        """Procesa un split completo (Train/Val/Test)."""
        logger.info(f"{'='*60}")
        logger.info(f"Ingeniería de Características para: {split_name} (Train={is_train})")
        logger.info(f"{'='*60}")
        
        # --- Preparación: Generar IDs únicos ---
        if 'unique_id' not in df_students.columns:
            df_students['unique_id'] = (df_students['id_student'].astype(str) + '_' + 
                                        df_students['code_module'] + '_' + 
                                        df_students['code_presentation'])

        for df in [df_assessments, df_interactions]:
            if 'unique_id' not in df.columns and 'id_student' in df.columns:
                df['unique_id'] = (df['id_student'].astype(str) + '_' + 
                                   df['code_module'] + '_' + 
                                   df['code_presentation'])
        
        # --- Preparación: Columna 'week' ---
        if 'week' not in df_interactions.columns:
            df_interactions['date'] = pd.to_numeric(df_interactions['date'], errors='coerce')
            df_interactions['week'] = (df_interactions['date'] // 7).astype('Int64')
        
        if 'week' not in df_assessments.columns:
            # Usar date_submitted para semana de entrega (no el deadline)
            col_date_for_week = 'date_submitted' if 'date_submitted' in df_assessments.columns else 'date'
            df_assessments[col_date_for_week] = pd.to_numeric(
                df_assessments[col_date_for_week], errors='coerce'
            )
            df_assessments['week'] = (df_assessments[col_date_for_week] // 7).astype('Int64')
        
        all_ids = df_students['unique_id'].unique()
        
        # =========================================================
        # 1. CLICS MULTICANAL (content, social, quiz, other)
        # =========================================================
        ts_clicks_log, ts_clicks_raw = self._generate_multichannel_clicks(df_interactions, all_ids)
        
        # Regularidad: Std Dev de clics TOTALES (para features estáticas)
        click_std = ts_clicks_raw.sum(axis=1).groupby(level=0).first()  # Total por estudiante
        # Recalcular std sobre la serie original de clics totales
        ts_total_clicks = self._create_week_buckets(df_interactions, 'sum_click', agg_func='sum', fill_val=0)
        ts_total_clicks = ts_total_clicks.reindex(all_ids, fill_value=0)
        click_std = ts_total_clicks.std(axis=1)
        click_std.name = 'click_std'
        
        # =========================================================
        # 2. RENDIMIENTO DERIVADO (avg, rate, trend × TMA, CMA)
        # =========================================================
        ts_performance = self._generate_performance_features(df_assessments, all_ids)
        
        # =========================================================
        # 3. PROCRASTINACIÓN (days_early)
        # =========================================================
        ts_procrastination = self._generate_procrastination(df_assessments, all_ids)
        
        # =========================================================
        # 4. FEATURES ESTÁTICAS + NORMALIZACIÓN
        # =========================================================
        X_static = self._generate_static_features(df_students, click_std, all_ids, is_train)
        
        # =========================================================
        # 5. TARGET
        # =========================================================
        # Mapeo Multiclase (4 Clases):
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
        
        # --- Resumen ---
        logger.info(f"Resumen de Features para '{split_name}':")
        logger.info(f"  ts_clicks:         {ts_clicks_log.shape}")
        logger.info(f"  ts_performance:    {ts_performance.shape}")
        logger.info(f"  ts_procrastination:{ts_procrastination.shape}")
        logger.info(f"  static_features:   {X_static.shape}")
        total = ts_clicks_log.shape[1] + ts_performance.shape[1] + ts_procrastination.shape[1] + X_static.shape[1]
        logger.info(f"  TOTAL FEATURES:    {total}")
        
        return {
            'ts_clicks': ts_clicks_log,
            'ts_performance': ts_performance,
            'ts_procrastination': ts_procrastination,
            'static_features': X_static,
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
=======
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import typer

from educational_ai_analytics.config import PROCESSED_DATA_DIR


# =========================================================================
# CONSTANTES DE DISEÑO
# =========================================================================
WEEK_START, WEEK_END = -2, 35  # Rango de semanas lectivas (40 semanas)

# Mapa de canales de actividad (basado en hallazgos del EDA)
# Fuente: docs/conclusiones_eda.md - "Calidad vs Cantidad (Perfil de Navegación)"
ACTIVITY_MAP = {
    'content': ['oucontent', 'resource', 'url', 'page', 'subpage'],  # Estudio pasivo
    'social': ['forumng', 'oucollaborate', 'ouwiki'],                # Aprendizaje social
    'quiz': ['quiz', 'questionnaire'],                               # Autoevaluación
    'other': []  # Placeholder: captura todo lo no mapeado (homepage, glossary, etc.)
}

# Tipos de evaluación a procesar (Exam excluido por data leakage)
ASSESSMENT_TYPES = ['TMA', 'CMA']


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

    # =====================================================================
    # HELPERS DE BUCKETING TEMPORAL
    # =====================================================================

    def _bucket_week_columns(self, ts_pivot, prefix, agg_func='sum', fill_val=0):
        """
        Aplica la lógica de bucketing semanal a un DataFrame ya pivotado.
        Agrupa semanas < WEEK_START en 'prev' y > WEEK_END en 'post'.
        
        Args:
            ts_pivot: DataFrame pivotado con semanas como columnas
            prefix: Prefijo para los nombres de columna (ej: 'content', 'TMA_avg')
            agg_func: 'sum' o 'mean' para agregar prev/post
            fill_val: Valor de relleno para semanas sin datos
        
        Returns:
            DataFrame con columnas: {prefix}_prev, {prefix}_w_neg2, ..., {prefix}_w_35, {prefix}_post
        """
        ts_buckets = pd.DataFrame(index=ts_pivot.index)
        available_weeks = sorted(ts_pivot.columns.tolist())
        
        cols_prev = [c for c in available_weeks if c < WEEK_START]
        cols_post = [c for c in available_weeks if c > WEEK_END]
        
        # Pre-Course
        if cols_prev:
            if fill_val == -1:
                # Para scores: ignorar -1 (no entregado) al agregar
                ts_buckets[f'{prefix}_w_prev'] = (
                    ts_pivot[cols_prev].replace(-1, np.nan).mean(axis=1).fillna(fill_val)
                )
            elif agg_func == 'sum':
                ts_buckets[f'{prefix}_w_prev'] = ts_pivot[cols_prev].sum(axis=1)
            else:
                ts_buckets[f'{prefix}_w_prev'] = ts_pivot[cols_prev].mean(axis=1)
        else:
            ts_buckets[f'{prefix}_w_prev'] = fill_val

        # Core Weeks (-2 a 35)
        for w in range(WEEK_START, WEEK_END + 1):
            col_name = f'{prefix}_w_{w}' if w >= 0 else f'{prefix}_w_neg{abs(w)}'
            if w in ts_pivot.columns:
                ts_buckets[col_name] = ts_pivot[w]
            else:
                ts_buckets[col_name] = fill_val

        # Post-Course
        if cols_post:
            if fill_val == -1:
                ts_buckets[f'{prefix}_w_post'] = (
                    ts_pivot[cols_post].replace(-1, np.nan).mean(axis=1).fillna(fill_val)
                )
            elif agg_func == 'sum':
                ts_buckets[f'{prefix}_w_post'] = ts_pivot[cols_post].sum(axis=1)
            else:
                ts_buckets[f'{prefix}_w_post'] = ts_pivot[cols_post].mean(axis=1)
        else:
            ts_buckets[f'{prefix}_w_post'] = fill_val
            
        return ts_buckets

    def _create_week_buckets(self, df_source, value_col, agg_func='sum', fill_val=0):
        """Genera matriz semanal dinámica (-2 a 35). Wrapper legacy."""
        if agg_func == 'sum':
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].sum().unstack(fill_value=fill_val)
        else:
            ts_pivot = df_source.groupby(['unique_id', 'week'])[value_col].mean().unstack(fill_value=fill_val)
        
        return self._bucket_week_columns(ts_pivot, value_col, agg_func, fill_val)

    # =====================================================================
    # GENERACIÓN DE CLICKS MULTICANAL
    # =====================================================================

    def _generate_multichannel_clicks(self, df_interactions, all_ids):
        """
        Genera matriz de clics desglosada por tipo de actividad.
        
        Canales: content, social, quiz, other
        Cada canal tiene 40 columnas (prev + weeks -2..35 + post)
        Total: 4 × 40 = 160 columnas
        
        Se aplica transformación log1p para normalizar la distribución power-law.
        """
        logger.info("- Generando Clics Multicanal (4 canales)...")
        channels_list = []
        
        # Obtener todos los activity_types mapeados para calcular 'other'
        all_mapped_types = sum(ACTIVITY_MAP.values(), [])
        
        for channel_name, target_types in ACTIVITY_MAP.items():
            logger.info(f"  · Canal: {channel_name} ({len(target_types)} tipos de actividad)")
            
            # Filtrar interacciones por tipo
            if target_types:
                df_sub = df_interactions[df_interactions['activity_type'].isin(target_types)]
            else:
                # Canal 'other': todo lo que NO está mapeado
                df_sub = df_interactions[~df_interactions['activity_type'].isin(all_mapped_types)]
            
            # Pivotar por semana
            if df_sub.empty:
                # Canal vacío: crear DataFrame con ceros
                ts_channel = pd.DataFrame(0, index=all_ids, 
                                          columns=[f'{channel_name}_w_prev'] +
                                                   [f'{channel_name}_w_{w}' if w >= 0 else f'{channel_name}_w_neg{abs(w)}' 
                                                    for w in range(WEEK_START, WEEK_END + 1)] +
                                                   [f'{channel_name}_w_post'])
            else:
                ts_pivot = df_sub.groupby(['unique_id', 'week'])['sum_click'].sum().unstack(fill_value=0)
                ts_channel = self._bucket_week_columns(ts_pivot, channel_name, agg_func='sum', fill_val=0)
            
            # Reindexar para alinear con todos los estudiantes
            ts_channel = ts_channel.reindex(all_ids, fill_value=0)
            channels_list.append(ts_channel)
        
        # Concatenar todos los canales
        ts_multichannel = pd.concat(channels_list, axis=1)
        
        # Log1p Transformation (normaliza distribución power-law de clics)
        ts_multichannel_log = np.log1p(ts_multichannel)
        
        logger.info(f"  → Matriz Clics Multicanal: {ts_multichannel_log.shape}")
        return ts_multichannel_log, ts_multichannel

    # =====================================================================
    # GENERACIÓN DE RENDIMIENTO DERIVADO (AVG, RATE, TREND)
    # =====================================================================

    def _get_performance_base_matrix(self, df_assessments, assess_type, all_ids):
        """
        Genera la matriz base de notas crudas para un tipo de evaluación.
        Usa -1 como sentinel para "no entregado/sin tarea en esa semana".
        """
        df_type = df_assessments[df_assessments['assessment_type'] == assess_type].copy()
        
        if df_type.empty:
            # No hay evaluaciones de este tipo
            cols = ([f'{assess_type}_w_prev'] +
                    [f'{assess_type}_w_{w}' if w >= 0 else f'{assess_type}_w_neg{abs(w)}'
                     for w in range(WEEK_START, WEEK_END + 1)] +
                    [f'{assess_type}_w_post'])
            return pd.DataFrame(-1, index=all_ids, columns=cols)
        
        # Pivotar: promedio de notas por semana (si hay varias entregas en la misma semana)
        ts_pivot = df_type.groupby(['unique_id', 'week'])['score'].mean().unstack(fill_value=-1)
        
        # Aplicar bucketing con -1 como fill_val
        ts_base = self._bucket_week_columns(ts_pivot, assess_type, agg_func='mean', fill_val=-1)
        ts_base = ts_base.reindex(all_ids, fill_value=-1)
        
        return ts_base

    def _generate_performance_features(self, df_assessments, all_ids):
        """
        Genera features derivadas de rendimiento para cada tipo de evaluación.
        
        Para cada tipo (TMA, CMA) se generan 3 métricas × 40 semanas:
          - avg:   Promedio acumulado (running average) normalizado 0-1
          - rate:  Tasa de entrega acumulada (submissions / total posible)
          - trend: Tendencia (nota actual - promedio histórico) normalizada
        
        Total: 2 tipos × 3 métricas × 40 semanas = 240 columnas
        
        Convención de valores:
          -1 = No hay evaluación de ese tipo en esa semana (sentinel)
          0  = Hubo evaluación pero no se entregó / nota 0
        """
        logger.info("- Generando Rendimiento Multicanal (avg, rate, trend × TMA, CMA)...")
        
        # Limpiar: solo registros con score y excluir Exam (data leakage)
        df_assess_clean = df_assessments.dropna(subset=['score']).copy()
        df_assess_clean['score'] = pd.to_numeric(df_assess_clean['score'], errors='coerce')
        df_assess_clean = df_assess_clean[df_assess_clean['assessment_type'].isin(ASSESSMENT_TYPES)]
        
        performance_matrices = []
        
        for assess_type in ASSESSMENT_TYPES:
            logger.info(f"  · Procesando canal de evaluación: {assess_type}...")
            
            # A. Matriz Base (Notas crudas con buckets, -1 = sin evaluación)
            ts_base = self._get_performance_base_matrix(df_assess_clean, assess_type, all_ids)
            ts_raw = ts_base.replace(-1, np.nan)  # NaN para cálculos
            cols = ts_base.columns
            
            # B. Feature 1: Promedio Acumulado (Running Average)
            # Expanding mean a lo largo del tiempo, normalizado 0-1
            # Forward fill de NaN con el promedio anterior
            feat_avg = ts_raw.T.expanding().mean().T.fillna(0) / 100.0
            feat_avg.columns = [c.replace(f'{assess_type}_', f'{assess_type}_avg_') for c in cols]
            
            # C. Feature 2: Tasa de Entrega (Submission Rate acumulada)
            # 1 si entregó, 0 si no
            is_submitted = (~ts_raw.isna()).astype(int)
            # Acumulado de entregas
            feat_rate = is_submitted.T.cumsum().T
            # Normalizar por el máximo de entregas vistas
            max_possible = feat_rate.max()
            feat_rate = feat_rate.div(max_possible.replace(0, 1), axis=1)
            feat_rate.columns = [c.replace(f'{assess_type}_', f'{assess_type}_rate_') for c in cols]
            
            # D. Feature 3: Tendencia (Score Trend)
            # Nota actual - Promedio histórico (0 si no entregó)
            # Permite detectar mejoras/empeoramientos progresivos
            trend_vals = (ts_raw.fillna(0).values - (feat_avg.values * 100)) / 100.0
            feat_trend = pd.DataFrame(
                trend_vals, 
                index=ts_base.index,
                columns=[c.replace(f'{assess_type}_', f'{assess_type}_trend_') for c in cols]
            )
            
            # Concatenar las 3 métricas de este canal
            performance_matrices.extend([feat_avg, feat_rate, feat_trend])
        
        # Consolidar todo
        ts_performance = pd.concat(performance_matrices, axis=1)
        
        logger.info(f"  → Matriz Rendimiento Multicanal: {ts_performance.shape}")
        return ts_performance

    # =====================================================================
    # GENERACIÓN DE PROCRASTINACIÓN
    # =====================================================================

    def _generate_procrastination(self, df_assessments, all_ids):
        """
        Genera serie temporal de procrastinación (days_early).
        
        days_early = date_deadline - date_submitted
          > 0: Entrega anticipada (bien)
          = 0: Entrega al límite
          < 0: Entrega tardía (riesgo)
          -999 (sentinel interno): Sin entrega → se reemplaza por 0
        """
        logger.info("- Generando Canal de Procrastinación (Days Early)...")
        
        df_assess_dates = df_assessments.dropna(subset=['date', 'date_submitted']).copy()
        df_assess_dates['date'] = pd.to_numeric(df_assess_dates['date'], errors='coerce')
        df_assess_dates['date_submitted'] = pd.to_numeric(df_assess_dates['date_submitted'], errors='coerce')
        df_assess_dates = df_assess_dates.dropna(subset=['date']).copy()
        df_assess_dates['days_early'] = df_assess_dates['date'] - df_assess_dates['date_submitted']
        
        # Pivotar con sentinel -999 para "sin entrega"
        ts_pivot = (df_assess_dates.groupby(['unique_id', 'week'])['days_early']
                    .mean().unstack(fill_value=-999))
        
        # Bucketing manual con reemplazo de sentinel
        ts_buckets = pd.DataFrame(index=ts_pivot.index)
        available_weeks = sorted(ts_pivot.columns.tolist())
        
        cols_prev = [c for c in available_weeks if c < WEEK_START]
        cols_core = [c for c in available_weeks if WEEK_START <= c <= WEEK_END]
        cols_post = [c for c in available_weeks if c > WEEK_END]
        
        # Pre-Course
        ts_buckets['days_early_w_prev'] = (
            ts_pivot[cols_prev].replace(-999, np.nan).mean(axis=1).fillna(0)
            if cols_prev else 0
        )
        
        # Core Weeks
        for w in range(WEEK_START, WEEK_END + 1):
            col_name = f'days_early_w_{w}' if w >= 0 else f'days_early_w_neg{abs(w)}'
            if w in ts_pivot.columns:
                # Reemplazar sentinel -999 por 0 (sin entrega = neutral)
                # Nota de Diseño: 0 puede significar "Entrega al límite" O "Sin Tarea"
                # Esto es aceptable porque el canal de Rendimiento (Scores) desambigua:
                # si Score != -1, entonces el 0 en procrastinación = "Entrega al límite"
                ts_buckets[col_name] = ts_pivot[w].replace(-999, 0)
            else:
                ts_buckets[col_name] = 0
        
        # Post-Course
        ts_buckets['days_early_w_post'] = (
            ts_pivot[cols_post].replace(-999, np.nan).mean(axis=1).fillna(0)
            if cols_post else 0
        )
        
        ts_procrastination = ts_buckets.reindex(all_ids, fill_value=0)
        logger.info(f"  → Matriz Procrastinación: {ts_procrastination.shape}")
        return ts_procrastination

    # =====================================================================
    # FEATURES ESTÁTICAS
    # =====================================================================

    def _generate_static_features(self, df_students, click_std, all_ids, is_train=False):
        """
        Genera features estáticas: demográficas, socioeconómicas, y regularidad.
        Aplica MinMaxScaler (fit on train, transform on val/test).
        """
        logger.info("- Generando Features Estáticas...")
        
        df_static = df_students.set_index('unique_id').copy()
        
        # --- Mapeos Ordinales ---
        df_static['imd_band_numeric'] = df_static['imd_band'].map(self.imd_map).fillna(-1)
        
        # Créditos
        if 'credits' not in df_static.columns:
            df_static['credits'] = 0
        df_static['credits'] = pd.to_numeric(df_static['credits'], errors='coerce').fillna(0)
        
        # Duración del módulo
        df_static['module_presentation_length'] = pd.to_numeric(
            df_static['module_presentation_length'], errors='coerce'
        ).fillna(0)
        
        # Fecha de registro (delay respecto al inicio del curso)
        df_static['date_registration'] = pd.to_numeric(
            df_static['date_registration'], errors='coerce'
        ).fillna(0)
        
        # --- Demográficos ---
        df_static['education_level'] = df_static['highest_education'].map(self.education_map).fillna(1)
        df_static['age_numeric'] = df_static['age_band'].map(self.age_map).fillna(0)
        df_static['gender_bool'] = df_static['gender'].map(self.gender_map).fillna(0)
        df_static['disability_bool'] = df_static['disability'].map(self.disability_map).fillna(0)
        
        # --- Selección de columnas ---
        static_cols = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 
            'education_level', 'age_numeric', 'gender_bool', 'disability_bool'
        ]
        X_static = df_static[static_cols].copy()
        
        # Añadir Regularidad (Click Std - varianza de clics totales)
        X_static = X_static.join(click_std).fillna(0)
        
        # --- Normalización (Fit on Train, Transform All) ---
        cols_to_scale = [
            'imd_band_numeric', 'credits', 'module_presentation_length', 
            'date_registration', 'num_of_prev_attempts', 'click_std',
            'education_level', 'age_numeric'
        ]
        cols_final = [c for c in cols_to_scale if c in X_static.columns]
        
        if is_train:
            logger.info("  * Entrenando Escaladores (MinMaxScaler 0-1)...")
            scaler = MinMaxScaler()
            X_static[cols_final] = scaler.fit_transform(X_static[cols_final])
            self.scalers['static'] = scaler
        else:
            logger.info("  * Aplicando Escaladores (Transform)...")
            if 'static' in self.scalers:
                scaler = self.scalers['static']
                X_static[cols_final] = scaler.transform(X_static[cols_final])
            else:
                logger.warning("No hay escalador entrenado. Usando datos crudos en Test/Val.")
        
        logger.info(f"  → Features Estáticas: {X_static.shape}")
        return X_static

    # =====================================================================
    # ORQUESTADOR PRINCIPAL
    # =====================================================================

    def process_split(self, split_name: str, df_students, df_assessments, df_interactions, is_train=False):
        """Procesa un split completo (Train/Val/Test)."""
        logger.info(f"{'='*60}")
        logger.info(f"Ingeniería de Características para: {split_name} (Train={is_train})")
        logger.info(f"{'='*60}")
        
        # --- Preparación: Generar IDs únicos ---
        if 'unique_id' not in df_students.columns:
            df_students['unique_id'] = (df_students['id_student'].astype(str) + '_' + 
                                        df_students['code_module'] + '_' + 
                                        df_students['code_presentation'])

        for df in [df_assessments, df_interactions]:
            if 'unique_id' not in df.columns and 'id_student' in df.columns:
                df['unique_id'] = (df['id_student'].astype(str) + '_' + 
                                   df['code_module'] + '_' + 
                                   df['code_presentation'])
        
        # --- Preparación: Columna 'week' ---
        if 'week' not in df_interactions.columns:
            df_interactions['date'] = pd.to_numeric(df_interactions['date'], errors='coerce')
            df_interactions['week'] = (df_interactions['date'] // 7).astype('Int64')
        
        if 'week' not in df_assessments.columns:
            # Usar date_submitted para semana de entrega (no el deadline)
            col_date_for_week = 'date_submitted' if 'date_submitted' in df_assessments.columns else 'date'
            df_assessments[col_date_for_week] = pd.to_numeric(
                df_assessments[col_date_for_week], errors='coerce'
            )
            df_assessments['week'] = (df_assessments[col_date_for_week] // 7).astype('Int64')
        
        all_ids = df_students['unique_id'].unique()
        
        # =========================================================
        # 1. CLICS MULTICANAL (content, social, quiz, other)
        # =========================================================
        ts_clicks_log, ts_clicks_raw = self._generate_multichannel_clicks(df_interactions, all_ids)
        
        # Regularidad: Std Dev de clics TOTALES (para features estáticas)
        click_std = ts_clicks_raw.sum(axis=1).groupby(level=0).first()  # Total por estudiante
        # Recalcular std sobre la serie original de clics totales
        ts_total_clicks = self._create_week_buckets(df_interactions, 'sum_click', agg_func='sum', fill_val=0)
        ts_total_clicks = ts_total_clicks.reindex(all_ids, fill_value=0)
        click_std = ts_total_clicks.std(axis=1)
        click_std.name = 'click_std'
        
        # =========================================================
        # 2. RENDIMIENTO DERIVADO (avg, rate, trend × TMA, CMA)
        # =========================================================
        ts_performance = self._generate_performance_features(df_assessments, all_ids)
        
        # =========================================================
        # 3. PROCRASTINACIÓN (days_early)
        # =========================================================
        ts_procrastination = self._generate_procrastination(df_assessments, all_ids)
        
        # =========================================================
        # 4. FEATURES ESTÁTICAS + NORMALIZACIÓN
        # =========================================================
        X_static = self._generate_static_features(df_students, click_std, all_ids, is_train)
        
        # =========================================================
        # 5. TARGET
        # =========================================================
        # Mapeo Multiclase (4 Clases):
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
        
        # --- Resumen ---
        logger.info(f"Resumen de Features para '{split_name}':")
        logger.info(f"  ts_clicks:         {ts_clicks_log.shape}")
        logger.info(f"  ts_performance:    {ts_performance.shape}")
        logger.info(f"  ts_procrastination:{ts_procrastination.shape}")
        logger.info(f"  static_features:   {X_static.shape}")
        total = ts_clicks_log.shape[1] + ts_performance.shape[1] + ts_procrastination.shape[1] + X_static.shape[1]
        logger.info(f"  TOTAL FEATURES:    {total}")
        
        return {
            'ts_clicks': ts_clicks_log,
            'ts_performance': ts_performance,
            'ts_procrastination': ts_procrastination,
            'static_features': X_static,
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
>>>>>>> c30dc9262eee8dc25b98d7ef8a910c40c00b5fda
