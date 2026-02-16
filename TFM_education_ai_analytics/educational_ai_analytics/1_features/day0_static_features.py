import numpy as np
import pandas as pd
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class DayZeroFeaturesBuilder:
    """
    Features estáticas/demográficas + prestart (día < 0).
    Mantiene estado (modas, region_map, OHE, course_stats) tras fit.
    """

    EPS = 1e-6
    P_CLIP = 0.995

    IMD_MAP = {
        "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
        "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
    }
    AGE_MAP = {"0-35": 0, "35-55": 1, "55<=": 2}
    EDUCATION_MAP = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4,
    }

    def __init__(self, ohe_transformer=None, normalize_mode: str = "zscore"):
        self.learned_stats: dict = {}
        self.region_map: dict = {}
        self.course_stats: dict = {}

        # ✅ FIX: atributo requerido por _apply_course_norm
        if normalize_mode not in {"zscore", "ratio_to_mean"}:
            raise ValueError("normalize_mode debe ser 'zscore' o 'ratio_to_mean'.")
        self.normalize_mode = normalize_mode

        from sklearn.preprocessing import OneHotEncoder
        if ohe_transformer is not None:
            self.ohe = ohe_transformer
        else:
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
            except TypeError:
                self.ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False)

    @staticmethod
    def _ensure_unique_id(df: pd.DataFrame) -> pd.DataFrame:
        if "unique_id" in df.columns:
            return df
        needed = {"id_student", "code_module", "code_presentation"}
        if not needed.issubset(df.columns):
            raise ValueError(f"Falta unique_id y no puedo construirlo (faltan {needed}).")
        out = df.copy()
        out["unique_id"] = (
            out["id_student"].astype(str) + "_"
            + out["code_module"].astype(str) + "_"
            + out["code_presentation"].astype(str)
        )
        return out

    def _clip_log1p(self, s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x = np.clip(x, 0.0, None)

        if x.size > 0:
            q = float(np.quantile(x, self.P_CLIP))
            if not np.isfinite(q) or q < 0:
                q = float(np.max(x)) if x.size else 0.0
            x = np.clip(x, 0.0, q)

        return pd.Series(np.log1p(x), index=s.index)

    def _get_course_series(self, df_students: pd.DataFrame, index_uids: pd.Index) -> pd.Series:
        s = self._ensure_unique_id(df_students.copy())
        s = s.drop_duplicates("unique_id").set_index("unique_id")
        if "code_module" in s.columns and "code_presentation" in s.columns:
            course = s["code_module"].astype(str) + "_" + s["code_presentation"].astype(str)
        else:
            course = pd.Series("UNKNOWN", index=s.index)
        return course.reindex(index_uids).fillna("UNKNOWN").astype(str)

    def _fit_course_stats(self, df: pd.DataFrame, course: pd.Series, cols: list[str]) -> dict:
        """
        Estructura:
        stats["__global__"][col] = {"mean": float, "std": float}
        stats[course_key][col] = {"mean": float, "std": float}
        """
        stats: dict = {}

        g_mu = df[cols].mean(axis=0)
        g_sd = df[cols].std(axis=0).replace(0, 1.0)

        stats["__global__"] = {c: {"mean": float(g_mu[c]), "std": float(g_sd[c])} for c in cols}

        for ckey, idxs in course.groupby(course).groups.items():
            block = df.loc[idxs, cols]
            mu = block.mean(axis=0)
            sd = block.std(axis=0).replace(0, 1.0)
            stats[str(ckey)] = {c: {"mean": float(mu[c]), "std": float(sd[c])} for c in cols}

        return stats

    @staticmethod
    def _get_mu_sd(block_stats: dict, col: str) -> tuple[float, float]:
        """
        Soporta:
        - Nuevo: {"mean": x, "std": y}
        - Legacy: float (mean) sin std
        """
        v = block_stats.get(col, None) if isinstance(block_stats, dict) else None

        if isinstance(v, dict):
            mu = float(v.get("mean", 0.0))
            sd = float(v.get("std", 1.0))
        elif v is None:
            mu, sd = 0.0, 1.0
        else:
            # legacy: era un float (mean) sin std
            mu, sd = float(v), 1.0

        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            sd = 1.0

        return mu, sd

    def _apply_course_norm(self, df: pd.DataFrame, course: pd.Series, stats: dict, cols: list[str]) -> pd.DataFrame:
        out = df.copy()

        # columnas float para escribir floats
        out[cols] = out[cols].astype(float)

        g = stats.get("__global__", {})
        global_mu = pd.Series({c: self._get_mu_sd(g, c)[0] for c in cols}, dtype=float)
        global_sd = pd.Series({c: self._get_mu_sd(g, c)[1] for c in cols}, dtype=float).replace(0, 1.0)

        for ckey, idxs in course.groupby(course).groups.items():
            key = str(ckey)
            local = stats.get(key, None)

            if isinstance(local, dict):
                mu = pd.Series({c: self._get_mu_sd(local, c)[0] for c in cols}, dtype=float)
                sd = pd.Series({c: self._get_mu_sd(local, c)[1] for c in cols}, dtype=float).replace(0, 1.0)
            else:
                mu, sd = global_mu, global_sd

            block = out.loc[idxs, cols].astype(float)

            if self.normalize_mode == "zscore":
                out.loc[idxs, cols] = (block - mu) / sd
            elif self.normalize_mode == "ratio_to_mean":
                out.loc[idxs, cols] = block / (mu + self.EPS)  # ✅ FIX: self.EPS
            else:
                raise ValueError("normalize_mode debe ser 'zscore' o 'ratio_to_mean'.")

        out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def extract_raw_features(
        self,
        df_students: pd.DataFrame,
        df_interactions: pd.DataFrame,
        index_uids: Optional[pd.Index] = None,
        *,
        day_col: str = "date",
        clicks_col: str = "sum_click",
    ) -> pd.DataFrame:
        stu = self._ensure_unique_id(df_students.copy())

        drop_bad = [c for c in ["final_result", "date_unregistration"] if c in stu.columns]
        if drop_bad:
            stu = stu.drop(columns=drop_bad)

        stu = stu.drop_duplicates(subset=["unique_id"]).set_index("unique_id")

        if index_uids is None:
            index_uids = stu.index
        index_uids = pd.Index(index_uids)

        keep_cols = [
            "code_module", "code_presentation", "gender", "region", "highest_education",
            "imd_band", "age_band", "disability", "num_of_prev_attempts", "studied_credits",
            "date_registration", "module_presentation_length",
        ]
        X_demo = stu[[c for c in keep_cols if c in stu.columns]].copy()

        for c in ["num_of_prev_attempts", "studied_credits", "date_registration", "module_presentation_length"]:
            if c in X_demo.columns:
                X_demo[c] = pd.to_numeric(X_demo[c], errors="coerce").fillna(0.0).astype(float)

        inter = self._ensure_unique_id(df_interactions.copy())

        if day_col not in inter.columns:
            raise ValueError(f"[interactions] falta columna '{day_col}'")

        inter["_day"] = pd.to_numeric(inter[day_col], errors="coerce").fillna(0.0).astype(float)

        if clicks_col in inter.columns:
            inter["_clicks"] = pd.to_numeric(inter[clicks_col], errors="coerce").fillna(0.0).astype(float)
        elif "clicks" in inter.columns:
            inter["_clicks"] = pd.to_numeric(inter["clicks"], errors="coerce").fillna(0.0).astype(float)
        else:
            raise ValueError(f"[interactions] falta '{clicks_col}' y 'clicks' para prestart")

        pre = inter[inter["_day"] < 0].copy()

        if pre.empty:
            agg = pd.DataFrame(columns=[
                "prestart_clicks_total", "prestart_active_days", "prestart_active_weeks", "prestart_earliest_day"
            ])
        else:
            pre["week"] = np.floor(pre["_day"] / 7.0).astype(int)
            agg = pre.groupby("unique_id").agg(
                prestart_clicks_total=("_clicks", "sum"),
                prestart_active_days=("_day", lambda s: int(pd.Series(s.dropna().astype(int)).nunique())),
                prestart_active_weeks=("week", lambda s: int(pd.Series(s.dropna().astype(int)).nunique())),
                prestart_earliest_day=("_day", "min"),
            )

        X_pre = agg.reindex(index=index_uids).fillna(0.0)

        X_pre["prestart_clicks_total"] = X_pre.get("prestart_clicks_total", 0.0).astype(float)
        X_pre["prestart_active_days"] = X_pre.get("prestart_active_days", 0).astype(int)
        X_pre["prestart_active_weeks"] = X_pre.get("prestart_active_weeks", 0).astype(int)
        X_pre["prestart_earliest_day"] = X_pre.get("prestart_earliest_day", 0.0).astype(float)

        X_pre["investigated_platform"] = (X_pre["prestart_clicks_total"] > 0).astype(int)
        X_pre["prestart_intensity"] = (
            X_pre["prestart_clicks_total"] / (X_pre["prestart_active_days"] + self.EPS)
        ).fillna(0.0)
        X_pre["prestart_anticipation"] = X_pre["prestart_earliest_day"].abs()

        out = X_demo.reindex(index=index_uids).join(X_pre, how="left").fillna(0.0)
        out.index.name = "unique_id"
        return out

    def process_pipeline(
        self,
        df_students: pd.DataFrame,
        df_interactions: pd.DataFrame,
        *,
        fit: bool = False,
        index_uids: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        df = self.extract_raw_features(df_students, df_interactions, index_uids=index_uids)

        # normaliza '?' -> NA
        for c in ["gender", "disability", "region", "highest_education", "imd_band", "age_band"]:
            if c in df.columns:
                df[c] = df[c].replace("?", pd.NA)

        # aprende modas y region_map SOLO en fit
        if fit:
            self.learned_stats["moda_age"] = (
                df["age_band"].mode().iloc[0] if ("age_band" in df.columns and not df["age_band"].mode().empty) else "0-35"
            )
            self.learned_stats["moda_imd"] = (
                df["imd_band"].mode().iloc[0] if ("imd_band" in df.columns and not df["imd_band"].mode().empty) else "50-60%"
            )
            if "region" in df.columns:
                unique_regions = sorted(df["region"].dropna().unique())
                self.region_map = {r: i for i, r in enumerate(unique_regions)}
            else:
                self.region_map = {}

        moda_imd = self.learned_stats.get("moda_imd", "50-60%")
        moda_age = self.learned_stats.get("moda_age", "0-35")

        # mappings
        df["imd_band"] = (
            df.get("imd_band", pd.Series(index=df.index, dtype=object))
            .fillna(moda_imd)
            .map(self.IMD_MAP).fillna(5).astype(int)
        )
        df["age_band"] = (
            df.get("age_band", pd.Series(index=df.index, dtype=object))
            .fillna(moda_age)
            .map(self.AGE_MAP).fillna(0).astype(int)
        )
        df["highest_education"] = (
            df.get("highest_education", pd.Series(index=df.index, dtype=object))
            .map(self.EDUCATION_MAP).fillna(1).astype(int)
        )
        df["region_encoded"] = (
            df.get("region", pd.Series(index=df.index, dtype=object))
            .map(self.region_map).fillna(-1).astype(int)
        )

        # OHE sobre categóricas
        cat_cols = ["gender", "disability"]
        for c in cat_cols:
            if c not in df.columns:
                df[c] = "Unknown"
            df[c] = df[c].fillna("Unknown").astype(str)

        ohe_values = self.ohe.fit_transform(df[cat_cols]) if fit else self.ohe.transform(df[cat_cols])
        o_cols = self.ohe.get_feature_names_out(cat_cols)
        ohe_cols = [re.sub(r"[^a-zA-Z0-9_]", "_", n.lower()) for n in o_cols]
        df_ohe = pd.DataFrame(ohe_values, columns=ohe_cols, index=df.index).astype("int8")

        base_cols = [
            "imd_band", "age_band", "highest_education",
            "num_of_prev_attempts", "studied_credits", "region_encoded",
            "prestart_clicks_total", "prestart_active_days", "prestart_active_weeks",
            "prestart_earliest_day", "investigated_platform",
            "prestart_intensity", "prestart_anticipation",
        ]

        for c in ["num_of_prev_attempts", "studied_credits"]:
            if c not in df.columns:
                df[c] = 0.0

        out_df = pd.concat([df[base_cols], df_ohe], axis=1)

        # columnas a normalizar por curso (todas las numéricas para evitar sesgos de escala)
        norm_cols = [
            "imd_band", "age_band", "highest_education", 
            "num_of_prev_attempts", "studied_credits", "region_encoded",
            "prestart_clicks_total", "prestart_active_days", "prestart_active_weeks", 
            "prestart_intensity", "prestart_anticipation", "prestart_earliest_day"
        ]

        # transform robusto ANTES de medias/std
        for c in ["prestart_clicks_total", "prestart_intensity"]:
            if c in out_df.columns:
                out_df[c] = self._clip_log1p(out_df[c])

        # course series para normalización
        course_series = self._get_course_series(df_students, out_df.index)

        # aprende stats SOLO en fit
        if fit:
            self.course_stats = self._fit_course_stats(out_df, course_series, norm_cols)

        # aplica stats si existen
        if self.course_stats:
            out_df = self._apply_course_norm(out_df, course_series, self.course_stats, norm_cols)

        # alineación final
        if index_uids is not None:
            out_df = out_df.reindex(pd.Index(index_uids)).fillna(0.0)

        out_df.index.name = "unique_id"
        return out_df


def build_day0_static_features(
    df_students: pd.DataFrame,
    df_interactions: pd.DataFrame,
    index_uids: Optional[pd.Index] = None,
    *,
    fit: bool = False,
    builder: Optional[DayZeroFeaturesBuilder] = None,
) -> pd.DataFrame:
    """
    Helper: usa process_pipeline (NO raw).
    Reusa builder entre splits para mantener estado.
    """
    if builder is None:
        builder = DayZeroFeaturesBuilder()
    return builder.process_pipeline(df_students, df_interactions, fit=fit, index_uids=index_uids)
