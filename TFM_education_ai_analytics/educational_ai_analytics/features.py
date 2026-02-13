import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import linregress

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

from educational_ai_analytics.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR

class FeatureEngineer:
    def __init__(self):
        # Mapeos ordinales
        self.imd_map = {
            '0-10%': 0, '10-20%': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
            '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9
        }
        self.age_map = {'0-35': 0, '35-55': 1, '55<=': 2}
        self.target_map = {'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 3}
        
        # Componentes de transformación
        self.scaler = StandardScaler()
        
        # Compatibilidad con scikit-learn < 1.2 y >= 1.2
        try:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.learned_stats = {}
        self.feature_names = None

    def _clean_column_names(self, names):
        """Normaliza nombres de columnas: minúsculas, sin espacios ni caracteres raros."""
        return [re.sub(r'[^a-zA-Z0-9_]', '_', name.replace(' ', '_')).lower() for name in names]

    def _prepare_unique_id(self, df):
        if 'unique_id' not in df.columns:
            df['unique_id'] = (df['id_student'].astype(str) + '_' + 
                              df['code_module'] + '_' + 
                              df['code_presentation'])
        return df

    def _process_demographics(self, df_students, fit=False):
        """Procesa variables demográficas con OHE y mapeos ordinales."""
        logger.info("- Procesando Demografía (OHE + Ordinal)...")
        df = df_students.copy()
        
        # 1. Imputación y Mapeo Ordinal
        if fit:
            self.learned_stats['moda_age'] = df['age_band'].mode()[0] if not df['age_band'].mode().empty else '0-35'
            self.learned_stats['moda_imd'] = df['imd_band'].mode()[0] if not df['imd_band'].mode().empty else '50-60%'
        
        df['imd_band'] = df['imd_band'].fillna(self.learned_stats.get('moda_imd', '50-60%')).map(self.imd_map).fillna(5)
        df['age_band'] = df['age_band'].fillna(self.learned_stats.get('moda_age', '0-35')).map(self.age_map).fillna(0)
        
        # 2. One-Hot Encoding
        cat_cols = ['gender', 'region', 'highest_education', 'disability']
        if fit:
            ohe_values = self.ohe.fit_transform(df[cat_cols])
        else:
            ohe_values = self.ohe.transform(df[cat_cols])
        
        clean_ohe_cols = self._clean_column_names(self.ohe.get_feature_names_out(cat_cols))
        df_ohe = pd.DataFrame(ohe_values, columns=clean_ohe_cols, index=df.index).astype('int8')
        
        # 3. Concatenar
        df_demo = pd.concat([
            df[['unique_id', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits']],
            df_ohe
        ], axis=1)
        
        return df_demo.drop_duplicates(subset=['unique_id']).set_index('unique_id')

    def _process_interactions(self, df_interactions, fit=False):
        """Procesa interacciones: absoluto, relativo al módulo y métricas de comportamiento."""
        logger.info("- Procesando Interacciones (Relativo + Behavioral Advanced)...")
        df = df_interactions.copy()
        
        # 1. Pivotar tipos de actividad
        df_clicks = df.groupby(['unique_id', 'activity_type'])['sum_click'].sum().unstack(fill_value=0)
        df_clicks.columns = [f'clicks_{col.lower()}' for col in df_clicks.columns]
        
        # 2. Engagement Ponderado
        weights = {
            'clicks_quiz': 3.0, 'clicks_subpage': 2.0, 'clicks_oucontent': 2.0,
            'clicks_forumng': 1.5, 'clicks_resource': 1.5, 'clicks_homepage': 1.0, 'clicks_url': 1.0
        }
        weighted_score = pd.Series(0, index=df_clicks.index)
        for col, weight in weights.items():
            if col in df_clicks.columns:
                weighted_score += df_clicks[col] * weight
        
        df_clicks['total_weighted_engagement'] = weighted_score
        df_clicks['total_raw_clicks'] = df_clicks.filter(like='clicks_').sum(axis=1)
        df_clicks['activity_diversity'] = (df_clicks.filter(like='clicks_') > 0).sum(axis=1)
        
        # 3. Métrica Relativa (Evitando Data Leakage usando estadísticas de cohorte)
        # Usamos merging o mapping vectorizado en lugar de .apply(axis=1) por rendimiento
        student_cohort_map = df[['unique_id', 'code_module', 'code_presentation']].drop_duplicates('unique_id').set_index('unique_id')
        df_clicks['temp_group'] = student_cohort_map['code_module'] + '_' + student_cohort_map['code_presentation']
        
        if fit:
            # Calculamos y guardamos estadísticas por cohorte (modulo + presentacion)
            cohort_stats = df_clicks.groupby('temp_group')['total_weighted_engagement'].agg(['mean', 'std']).to_dict('index')
            self.learned_stats['cohort_engagement'] = cohort_stats
            # Fallback global para cohortes que no estén en train
            self.learned_stats['global_engagement'] = {
                'mean': df_clicks['total_weighted_engagement'].mean(),
                'std': df_clicks['total_weighted_engagement'].std()
            }
            
        # Aplicamos normalización vectorizada usando lo aprendido en Train
        cohort_map = self.learned_stats.get('cohort_engagement', {})
        global_stats = self.learned_stats.get('global_engagement', {'mean': 0, 'std': 1})
        
        # Mapeo de medias y desviaciones (vectorizado)
        means = df_clicks['temp_group'].map(lambda x: cohort_map.get(x, {}).get('mean', global_stats['mean']))
        stds = df_clicks['temp_group'].map(lambda x: cohort_map.get(x, {}).get('std', global_stats['std']))
        
        df_clicks['rel_eng_zscore'] = (df_clicks['total_weighted_engagement'] - means) / stds.replace(0, 1)
        df_clicks.drop(columns=['temp_group'], inplace=True)
        
        # 4. Métricas Avanzadas (Aceleración, Weekend, Curiosidad)
        # 4.1 Effort Slope
        df['week'] = np.floor(df['date'] / 7)
        weekly = df.groupby(['unique_id', 'week'])['sum_click'].sum().reset_index()
        def calc_slope(g):
            if len(g) < 2 or g['week'].nunique() <= 1: return 0.0
            slope, _, _, _, _ = linregress(g['week'], g['sum_click'])
            return slope if not np.isnan(slope) else 0.0
        effort_slope = weekly.groupby('unique_id').apply(calc_slope).rename('effort_slope')
        
        # 4.2 Weekend Ratio
        df['is_weekend'] = (df['date'] % 7).isin([5, 6])
        weekend_clicks = df.groupby(['unique_id', 'is_weekend'])['sum_click'].sum().unstack(fill_value=0)
        if True not in weekend_clicks.columns: weekend_clicks[True] = 0
        weekend_ratio = (weekend_clicks[True] / (weekend_clicks.sum(axis=1) + 1e-5)).rename('weekend_ratio')
        
        # 4.3 Curiosity Index
        aux_acts = ['glossary', 'oucollaborate', 'resource', 'forumng', 'dataplus']
        aux_clicks = df[df['activity_type'].isin(aux_acts)].groupby('unique_id')['sum_click'].sum()
        curiosity_index = (aux_clicks / (df_clicks['total_raw_clicks'] + 1e-5)).fillna(0).rename('curiosity_index')
        
        adv_features = pd.concat([effort_slope, weekend_ratio, curiosity_index], axis=1).fillna(0)
        return df_clicks.join(adv_features, how='left').fillna(0)

    def _process_performance(self, df_assessments):
        """Procesa rendimiento con tendencia (slope) y fiabilidad de entrega."""
        logger.info("- Procesando Rendimiento (Slope + Pass Ratio)...")
        df = df_assessments[df_assessments['assessment_type'] != 'Exam'].copy()
        df.sort_values(by=['unique_id', 'date'], inplace=True)
        
        for col in ['score', 'date_submitted', 'date', 'weight']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        df['weighted_score'] = df['score'] * (df['weight'] / 100.0)
        df['is_late'] = (df['date_submitted'] > df['date']).astype(int)
        df['is_passed'] = (df['score'] >= 40).astype(int)
        
        def calc_slope(x):
            if len(x) < 2: return 0.0
            s, _, _, _, _ = linregress(range(len(x)), x)
            return s if not np.isnan(s) else 0.0

        # 4. Agregación (Segura con Named Aggregation)
        perf = df.groupby('unique_id').agg(
            avg_score=('score', 'mean'),
            score_std=('score', 'std'),
            score_slope=('score', calc_slope),
            api_index=('weighted_score', 'sum'),
            late_ratio=('is_late', 'mean'),
            pass_ratio=('is_passed', 'mean'),
            submission_count=('id_assessment', 'count')
        )
        perf['has_submitted'] = 1
        return perf.fillna(0)

    def process_split(self, df_students, df_assessments, df_interactions, fit=False):
        """Une bloques, aplica Log-Scaling selectivo y realiza el escalado final único."""
        # Aseguramos ID (retornando dfs para evitar problemas de side-effects)
        df_students = self._prepare_unique_id(df_students)
        df_assessments = self._prepare_unique_id(df_assessments)
        df_interactions = self._prepare_unique_id(df_interactions)
        
        X_demo = self._process_demographics(df_students, fit=fit)
        X_inter = self._process_interactions(df_interactions, fit=fit)
        X_perf = self._process_performance(df_assessments)
        
        X = X_demo.join(X_inter, how='left').join(X_perf, how='left')
        X['has_submitted'] = X['has_submitted'].fillna(0)
        X = X.fillna(0)
        
        if fit:
            self.feature_names = X.columns.tolist()
        elif self.feature_names is not None:
            X = X.reindex(columns=self.feature_names, fill_value=0)
            
        # Log-Scaling a intensidades/conteos (Excluimos Z-scores, Slopes y Ratios)
        cols_log = [c for c in X.columns if any(k in c for k in ['clicks', 'weighted', 'api', 'count']) 
                    and not any(k in c for k in ['zscore', 'slope', 'ratio'])]
        X[cols_log] = np.log1p(X[cols_log].astype(float))
        
        # Escalado Final Único
        if fit:
            X_scaled_arr = self.scaler.fit_transform(X)
        else:
            X_scaled_arr = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled_arr, index=X.index, columns=X.columns), X

def run_feature_pipeline():
    # Limpiar directorio de salida para evitar mezclar datos de ejecuciones anteriores
    if FEATURES_DATA_DIR.exists():
        import shutil
        logger.info(f"🧹 Limpiando directorio de features: {FEATURES_DATA_DIR}")
        shutil.rmtree(FEATURES_DATA_DIR)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    engineer = FeatureEngineer()
    
    for split in ['training', 'validation', 'test']:
        logger.info(f"\n🚀 GENERANDO FEATURES PARA: {split.upper()}")
        split_path = PROCESSED_DATA_DIR / split
        
        try:
            # Cargar y preparar IDs explícitamente
            dfs = {}
            for f in ['students', 'assessments', 'interactions']:
                df = pd.read_csv(split_path / f"{f}.csv")
                dfs[f] = engineer._prepare_unique_id(df)
        except FileNotFoundError:
            logger.error(f"Faltan archivos en {split_path}. Ejecuta 'make data' primero.")
            return

        X_scaled, X_raw = engineer.process_split(dfs['students'], dfs['assessments'], dfs['interactions'], fit=(split == 'training'))
        
        out_dir = FEATURES_DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        X_scaled.to_csv(out_dir / "static_features.csv")
        X_raw.to_csv(out_dir / "raw_features.csv")
        
        target = dfs['students'].set_index('unique_id').loc[X_scaled.index, ['final_result']]
        target['final_result'] = target['final_result'].map(engineer.target_map).fillna(0).astype(int)
        target.to_csv(out_dir / "target.csv")
        logger.info(f"✅ OK: {len(X_scaled)} registros en {out_dir}")

if __name__ == "__main__":
    run_feature_pipeline()