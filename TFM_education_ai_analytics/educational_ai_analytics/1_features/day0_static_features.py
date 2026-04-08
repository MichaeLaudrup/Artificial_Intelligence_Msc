import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DayZeroFeaturesBuilder:
    """
    Features estáticas/demográficas + prestart (día < 0).
    Mantiene estado (modas, region_map, course_stats) tras fit.
    """

    EPS = 1e-6
    P_CLIP = 0.99

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

    def __init__(self, normalize_mode: str = "zscore"):
        self.learned_stats: dict = {}
        self.region_map: dict = {}
        self.course_stats: dict = {}

        if normalize_mode not in {"zscore", "ratio_to_mean"}:
            raise ValueError("normalize_mode debe ser 'zscore' o 'ratio_to_mean'.")
        self.normalize_mode = normalize_mode

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
        s = df_students.set_index("unique_id")
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
        v = block_stats.get(col, {"mean": 0.0, "std": 1.0})
        return float(v["mean"]), float(v["std"])

    def _apply_course_norm(self, df: pd.DataFrame, course: pd.Series, stats: dict, cols: list[str]) -> pd.DataFrame:
        out = df.copy().astype(float)
        g = stats.get("__global__", {})

        for ckey, idxs in course.groupby(course).groups.items():
            local = stats.get(str(ckey), g)
            mu = pd.Series({c: self._get_mu_sd(local, c)[0] for c in cols})
            sd = pd.Series({c: self._get_mu_sd(local, c)[1] for c in cols}).replace(0, 1.0)

            block = out.loc[idxs, cols]
            if self.normalize_mode == "zscore":
                out.loc[idxs, cols] = (block - mu) / sd
            else:
                out.loc[idxs, cols] = block / (mu + self.EPS)

        return out.fillna(0.0)

    def extract_raw_features(
        self,
        df_students: pd.DataFrame,
        df_interactions: pd.DataFrame,
        index_uids: Optional[pd.Index]
    ) -> pd.DataFrame:
        stu = df_students.drop_duplicates(subset=["unique_id"]).set_index("unique_id")

        drop_bad = [c for c in ["final_result", "date_unregistration"] if c in stu.columns]
        if drop_bad:
            stu = stu.drop(columns=drop_bad)

        if index_uids is None:
            index_uids = stu.index
        index_uids = pd.Index(index_uids)

        keep_cols = [
            "code_module", "code_presentation", "region", "highest_education",
            "imd_band", "age_band", "num_of_prev_attempts", "studied_credits",
            "date_registration", "module_presentation_length",
        ]
        X_demo = stu[keep_cols].copy()

        # NOTE: 'date_registration' already comes imputed via median from 0_dataset.py.
        # We use .fillna(0.0) here only as a safety fallback for residual nulls (e.g. new courses 
        # not seen in training) to keep the pipeline robust, not as the primary strategy.
        for c in ["num_of_prev_attempts", "studied_credits", "date_registration", "module_presentation_length"]:
            X_demo[c] = pd.to_numeric(X_demo[c], errors="coerce").fillna(0.0).astype(float)

        inter = df_interactions.copy()
        pre = inter[inter["date"] < 0].copy()

        if pre.empty:
            X_pre = pd.DataFrame(0.0, index=index_uids, columns=[
                "prestart_clicks_total", "prestart_active_days",
                "prestart_active_weeks", "prestart_earliest_day",
            ])
        else:
            pre["week"] = np.floor(pre["date"] / 7.0).astype(int)
            X_pre = (
                pre.groupby("unique_id")
                .agg(
                    prestart_clicks_total=("sum_click", "sum"),
                    prestart_active_days=("date", "nunique"),
                    prestart_active_weeks=("week", "nunique"),
                    prestart_earliest_day=("date", "min"),
                )
                .reindex(index_uids)
                .fillna(0.0)
            )

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

        for c in ["region", "highest_education", "imd_band", "age_band"]:
            df[c] = df[c].replace("?", pd.NA)

        if fit:
            self.learned_stats["moda_age"] = df["age_band"].mode().iloc[0] if not df["age_band"].mode().empty else "0-35"
            self.learned_stats["moda_imd"] = df["imd_band"].mode().iloc[0] if not df["imd_band"].mode().empty else "50-60%"
            unique_regions = sorted(df["region"].dropna().unique())
            self.region_map = {r: i for i, r in enumerate(unique_regions)}

        moda_imd = self.learned_stats.get("moda_imd", "50-60%")
        moda_age = self.learned_stats.get("moda_age", "0-35")

        df["imd_band"] = df["imd_band"].fillna(moda_imd).map(self.IMD_MAP).fillna(5).astype(int)
        df["age_band"] = df["age_band"].fillna(moda_age).map(self.AGE_MAP).fillna(0).astype(int)
        df["highest_education"] = df["highest_education"].map(self.EDUCATION_MAP).fillna(1).astype(int)
        df["region_encoded"] = df["region"].map(self.region_map).fillna(-1).astype(int)

        base_cols = [
            "imd_band", "age_band", "highest_education",
            "num_of_prev_attempts", "studied_credits", "date_registration", "region_encoded",
            "prestart_clicks_total", "prestart_active_days", "prestart_active_weeks",
            "prestart_earliest_day", "investigated_platform",
            "prestart_intensity", "prestart_anticipation",
        ]

        out_df = df[base_cols].copy()

        norm_cols = [
            "imd_band", "age_band", "highest_education", 
            "num_of_prev_attempts", "studied_credits", "date_registration", "region_encoded",
            "prestart_clicks_total", "prestart_active_days", "prestart_active_weeks", 
            "prestart_intensity", "prestart_anticipation", "prestart_earliest_day"
        ]

        for c in ["prestart_clicks_total", "prestart_intensity"]:
            out_df[c] = self._clip_log1p(out_df[c])

        course_series = self._get_course_series(df_students, out_df.index)

        if fit:
            self.course_stats = self._fit_course_stats(out_df, course_series, norm_cols)

        if self.course_stats:
            out_df = self._apply_course_norm(out_df, course_series, self.course_stats, norm_cols)

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
    """Runs the day-0 feature pipeline and reuses the builder state if provided."""
    if builder is None:
        builder = DayZeroFeaturesBuilder()
    return builder.process_pipeline(df_students, df_interactions, fit=fit, index_uids=index_uids)
