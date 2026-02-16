"""
Simplificado:
- raw_features.csv      -> features construidas (join + métricas), SIN log1p y SIN StandardScaler
- engineered_features.csv -> features finales para modelar: (sparsity-drop + log1p + StandardScaler)
- target.csv            -> etiqueta mapeada a int

No se guardan "static_features" ni ficheros intermedios extra.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from educational_ai_analytics.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR

# Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        # Ordinal maps
        self.imd_map = {
            "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
            "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
        }
        self.age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
        self.education_map = {
            "No Formal quals": 0,
            "Lower Than A Level": 1,
            "A Level or Equivalent": 2,
            "HE Qualification": 3,
            "Post Graduate Qualification": 4,
        }
        self.target_map = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}

        # Transformers
        self.scaler = StandardScaler()

        # OHE: intenta drop de binarias para no duplicar (si no, fallback)
        ohe_kwargs = dict(handle_unknown="ignore")
        try:
            try:
                self.ohe = OneHotEncoder(drop="if_binary", sparse_output=False, **ohe_kwargs)
            except TypeError:
                self.ohe = OneHotEncoder(drop="if_binary", sparse=False, **ohe_kwargs)
        except TypeError:
            try:
                self.ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
            except TypeError:
                self.ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

        # Learned only on train
        self.learned_stats = {}
        self.region_map = {}
        self.feature_names = None
        self.cols_to_drop = []

        # Robust params
        self.ZSCORE_CLIP = 5.0
        self.EPS = 1e-6

    @staticmethod
    def _clean_column_names(names):
        return [re.sub(r"[^a-zA-Z0-9_]", "_", n.replace(" ", "_")).lower() for n in names]

    @staticmethod
    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _prepare_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if "unique_id" not in df.columns:
            df = df.copy()
            df["unique_id"] = (
                df["id_student"].astype(str)
                + "_"
                + df["code_module"].astype(str)
                + "_"
                + df["code_presentation"].astype(str)
            )
        return df

    @staticmethod
    def _safe_slope(x: pd.Series, y: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = x.notna() & y.notna()
        x = x[m]
        y = y[m]
        if len(x) < 2 or x.nunique() <= 1:
            return 0.0
        slope, _, _, _, _ = linregress(x, y)
        if np.isnan(slope) or np.isinf(slope):
            return 0.0
        return float(slope)

    # -------------------- BLOCKS --------------------

    def _process_demographics(self, df_students: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        logger.info("- Procesando Demografía...")
        df = df_students.copy()

        for c in ["gender", "disability", "region", "highest_education", "imd_band", "age_band"]:
            if c in df.columns:
                df[c] = df[c].replace("?", pd.NA)

        if fit:
            self.learned_stats["moda_age"] = df["age_band"].mode().iloc[0] if not df["age_band"].mode().empty else "0-35"
            self.learned_stats["moda_imd"] = df["imd_band"].mode().iloc[0] if not df["imd_band"].mode().empty else "50-60%"

        df["imd_band"] = (
            df["imd_band"].fillna(self.learned_stats.get("moda_imd", "50-60%")).map(self.imd_map).fillna(5).astype(int)
        )
        df["age_band"] = (
            df["age_band"].fillna(self.learned_stats.get("moda_age", "0-35")).map(self.age_map).fillna(0).astype(int)
        )
        df["highest_education"] = df["highest_education"].map(self.education_map).fillna(1).astype(int)

        for c in ["num_of_prev_attempts", "studied_credits"]:
            if c in df.columns:
                df[c] = self._to_num(df[c]).fillna(0).astype(float)

        if fit:
            unique_regions = sorted(df["region"].dropna().unique())
            self.region_map = {r: i for i, r in enumerate(unique_regions)}

        df["region_encoded"] = df["region"].map(self.region_map).fillna(-1).astype(int)

        cat_cols = ["gender", "disability"]
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown").astype(str)

        if fit:
            ohe_values = self.ohe.fit_transform(df[cat_cols])
        else:
            ohe_values = self.ohe.transform(df[cat_cols])

        ohe_cols = self._clean_column_names(self.ohe.get_feature_names_out(cat_cols))
        df_ohe = pd.DataFrame(ohe_values, columns=ohe_cols, index=df.index).astype("int8")

        base_cols = [
            "unique_id",
            "imd_band",
            "age_band",
            "highest_education",
            "num_of_prev_attempts",
            "studied_credits",
            "region_encoded",
        ]
        out = pd.concat([df[base_cols], df_ohe], axis=1)
        return out.drop_duplicates(subset=["unique_id"]).set_index("unique_id")

    def _process_interactions(self, df_interactions: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        logger.info("- Procesando Interacciones...")
        df = df_interactions.copy()

        df["sum_click"] = self._to_num(df["sum_click"]).fillna(0).astype(float)
        df["date"] = self._to_num(df["date"]).fillna(0).astype(float)

        # 1) pivot activity_type -> clicks_*
        df_clicks = df.groupby(["unique_id", "activity_type"])["sum_click"].sum().unstack(fill_value=0)
        df_clicks.columns = [f"clicks_{str(c).lower()}" for c in df_clicks.columns]
        if "clicks_repeatactivity" in df_clicks.columns:
            df_clicks = df_clicks.drop(columns=["clicks_repeatactivity"])

        # 2) engagement + basics
        weights = {
            "clicks_quiz": 3.0,
            "clicks_subpage": 2.0,
            "clicks_oucontent": 2.0,
            "clicks_forumng": 1.5,
            "clicks_resource": 1.5,
            "clicks_homepage": 1.0,
            "clicks_url": 1.0,
        }
        weighted = pd.Series(0.0, index=df_clicks.index)
        for col, w in weights.items():
            if col in df_clicks.columns:
                weighted = weighted + df_clicks[col].astype(float) * float(w)

        df_clicks["total_weighted_engagement"] = weighted
        df_clicks["total_raw_clicks"] = df_clicks.filter(like="clicks_").sum(axis=1).astype(float)
        df_clicks["activity_diversity"] = (df_clicks.filter(like="clicks_") > 0).sum(axis=1).astype(int)

        # 3) relative zscore by cohort (module+presentation) learned on train
        cohort = (
            df[["unique_id", "code_module", "code_presentation"]]
            .drop_duplicates("unique_id")
            .set_index("unique_id")
        )
        df_clicks["temp_group"] = cohort["code_module"].astype(str) + "_" + cohort["code_presentation"].astype(str)

        if fit:
            cohort_stats = (
                df_clicks.groupby("temp_group")["total_weighted_engagement"]
                .agg(["mean", "std"])
                .to_dict("index")
            )
            self.learned_stats["cohort_engagement"] = cohort_stats
            self.learned_stats["global_engagement"] = {
                "mean": float(df_clicks["total_weighted_engagement"].mean()),
                "std": float(df_clicks["total_weighted_engagement"].std()),
            }

        cohort_map = self.learned_stats.get("cohort_engagement", {})
        global_stats = self.learned_stats.get("global_engagement", {"mean": 0.0, "std": 1.0})

        mean_map = {k: v.get("mean", global_stats["mean"]) for k, v in cohort_map.items()}
        std_map = {k: v.get("std", global_stats["std"]) for k, v in cohort_map.items()}

        means = df_clicks["temp_group"].map(mean_map).fillna(global_stats["mean"]).astype(float)
        stds = df_clicks["temp_group"].map(std_map).fillna(global_stats["std"]).astype(float).replace(0, 1.0)

        rel = (df_clicks["total_weighted_engagement"] - means) / stds
        df_clicks["rel_eng_zscore"] = rel.clip(-self.ZSCORE_CLIP, self.ZSCORE_CLIP)
        df_clicks = df_clicks.drop(columns=["temp_group"])

        # 4) advanced, robust (no weekend)
        df["week"] = np.floor(df["date"] / 7.0).astype(int)
        weekly = df.groupby(["unique_id", "week"])["sum_click"].sum().reset_index()

        def weekly_slope(g):
            if len(g) < 2 or g["week"].nunique() <= 1:
                return 0.0
            slope, _, _, _, _ = linregress(g["week"], g["sum_click"])
            if np.isnan(slope) or np.isinf(slope):
                return 0.0
            return float(slope)

        effort_slope = weekly.groupby("unique_id").apply(weekly_slope).rename("effort_slope").clip(-20, 20)
        active_weeks = weekly[weekly["sum_click"] > 0].groupby("unique_id")["week"].nunique().rename("active_weeks").fillna(0).astype(int)

        total_clicks = df.groupby("unique_id")["sum_click"].sum().astype(float)
        prestart_clicks = df[df["date"] < 0].groupby("unique_id")["sum_click"].sum().astype(float)
        prestart_ratio = (prestart_clicks / (total_clicks + self.EPS)).fillna(0).rename("prestart_ratio")

        early_clicks = df[(df["week"] >= 0) & (df["week"] <= 3)].groupby("unique_id")["sum_click"].sum().astype(float)
        early_weeks_ratio = (early_clicks / (total_clicks + self.EPS)).fillna(0).rename("early_weeks_ratio")

        aux_acts = ["glossary", "oucollaborate", "resource", "forumng", "dataplus"]
        aux_clicks = df[df["activity_type"].isin(aux_acts)].groupby("unique_id")["sum_click"].sum().astype(float)
        curiosity_index = (aux_clicks / (df_clicks["total_raw_clicks"] + self.EPS)).fillna(0).rename("curiosity_index")

        adv = pd.concat([effort_slope, active_weeks, prestart_ratio, early_weeks_ratio, curiosity_index], axis=1).fillna(0)
        return df_clicks.join(adv, how="left").fillna(0)

    def _process_performance(self, df_assessments: pd.DataFrame) -> pd.DataFrame:
        logger.info("- Procesando Rendimiento...")
        df = df_assessments.copy()
        df = df[df["assessment_type"] != "Exam"].copy()

        for c in ["score", "date_submitted", "date", "weight"]:
            df[c] = self._to_num(df[c])

        df["score"] = df["score"].fillna(0).astype(float)
        df["date_submitted"] = df["date_submitted"].fillna(0).astype(float)
        df["date"] = df["date"].fillna(0).astype(float)
        df["weight"] = df["weight"].fillna(0).astype(float)

        df.sort_values(by=["unique_id", "date"], inplace=True)

        df["weighted_score"] = df["score"] * (df["weight"] / 100.0)
        df["is_late"] = (df["date_submitted"] > df["date"]).astype(int)
        df["is_passed"] = (df["score"] >= 40).astype(int)

        score_slope = df.groupby("unique_id").apply(lambda g: self._safe_slope(g["date"], g["score"])).rename("score_slope")

        perf = df.groupby("unique_id").agg(
            avg_score=("score", "mean"),
            score_std=("score", "std"),
            api_index=("weighted_score", "sum"),
            late_ratio=("is_late", "mean"),
            pass_ratio=("is_passed", "mean"),
            submission_count=("id_assessment", "count"),
        )
        perf = perf.join(score_slope, how="left")
        perf["has_submitted"] = 1
        return perf.fillna(0)

    # -------------------- ASSEMBLY --------------------

    def _build_raw_X(self, df_students, df_assessments, df_interactions, fit: bool):
        """Construye X sin log1p ni StandardScaler (pero con las métricas ya creadas)."""
        df_students = self._prepare_unique_id(df_students)
        df_assessments = self._prepare_unique_id(df_assessments)
        df_interactions = self._prepare_unique_id(df_interactions)

        if df_students["unique_id"].duplicated().any():
            ex = df_students[df_students["unique_id"].duplicated(keep=False)]["unique_id"].head(5).tolist()
            raise ValueError(f"students tiene unique_id duplicados. Ejemplos: {ex}")

        X_demo = self._process_demographics(df_students, fit=fit)
        X_inter = self._process_interactions(df_interactions, fit=fit)
        X_perf = self._process_performance(df_assessments)

        X = X_demo.join(X_inter, how="left").join(X_perf, how="left")
        X["has_submitted"] = X["has_submitted"].fillna(0)
        return X.fillna(0)

    def _transform_to_engineered(self, X_raw: pd.DataFrame, fit: bool):
        """Aplica TODO lo 'final' para modelar: sparsity-drop + log1p + StandardScaler."""
        X = X_raw.copy()

        # Filtro sparsity (solo fit en train)
        if fit:
            sparsity = (X == 0).mean()
            self.cols_to_drop = sparsity[sparsity > 0.95].index.tolist()
            if self.cols_to_drop:
                logger.info(f"⚠️ Eliminando {len(self.cols_to_drop)} columnas con >95% ceros.")

        if self.cols_to_drop:
            drop_now = [c for c in self.cols_to_drop if c in X.columns]
            if drop_now:
                X = X.drop(columns=drop_now)

        # Congelar espacio/orden de features
        if fit:
            self.feature_names = X.columns.tolist()
        else:
            if self.feature_names is None:
                raise RuntimeError("feature_names no inicializado. Ejecuta primero el split training.")
            X = X.reindex(columns=self.feature_names, fill_value=0)

        # Log1p a conteos/intensidades (no zscores/slopes/ratios)
        cols_log = [
            c for c in X.columns
            if any(k in c for k in ["clicks", "weighted", "api", "count"])
            and not any(k in c for k in ["zscore", "slope", "ratio"])
        ]
        if cols_log:
            X.loc[:, cols_log] = np.log1p(X[cols_log].astype(float))

        # StandardScaler
        if fit:
            arr = self.scaler.fit_transform(X)
        else:
            arr = self.scaler.transform(X)

        return pd.DataFrame(arr, index=X.index, columns=X.columns)

    def process_split(self, df_students, df_assessments, df_interactions, fit: bool):
        """Devuelve (X_raw, X_engineered_scaled)."""
        X_raw = self._build_raw_X(df_students, df_assessments, df_interactions, fit=fit)
        X_eng = self._transform_to_engineered(X_raw, fit=fit)
        return X_raw, X_eng


def run_feature_pipeline():
    # Limpieza de salida
    if FEATURES_DATA_DIR.exists():
        import shutil
        logger.info(f"🧹 Limpiando directorio de features: {FEATURES_DATA_DIR}")
        shutil.rmtree(FEATURES_DATA_DIR)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    engineer = FeatureEngineer()

    for split in ["training", "validation", "test"]:
        logger.info(f"\n🚀 GENERANDO FEATURES PARA: {split.upper()}")
        split_path = PROCESSED_DATA_DIR / split

        try:
            dfs = {}
            for f in ["students", "assessments", "interactions"]:
                df = pd.read_csv(split_path / f"{f}.csv")
                dfs[f] = engineer._prepare_unique_id(df)
        except FileNotFoundError:
            logger.error(f"Faltan archivos en {split_path}. Ejecuta 'make data' primero.")
            return

        X_raw, X_eng = engineer.process_split(
            dfs["students"], dfs["assessments"], dfs["interactions"], fit=(split == "training")
        )

        out_dir = FEATURES_DATA_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Guardar solo 2: raw y engineered
        X_raw.to_csv(out_dir / "raw_features.csv")
        X_eng.to_csv(out_dir / "engineered_features.csv")

        # Target alineado al índice de X_eng (y por construcción también X_raw)
        students_idx = dfs["students"].set_index("unique_id")
        target = students_idx.loc[X_eng.index, ["final_result"]].copy()
        target["final_result"] = target["final_result"].map(engineer.target_map).fillna(0).astype(int)
        target.to_csv(out_dir / "target.csv")

        logger.info(f"✅ OK: {len(X_eng)} registros en {out_dir}")
        logger.info(f"📊 Features raw: {X_raw.shape[1]} | engineered: {X_eng.shape[1]}")

    logger.info("Pipeline de features completado.")


if __name__ == "__main__":
    run_feature_pipeline()
