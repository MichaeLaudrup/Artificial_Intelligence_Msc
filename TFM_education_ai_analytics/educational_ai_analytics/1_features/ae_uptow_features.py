import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

EPS = 1e-6


@dataclass
class AEUptoWFeaturesBuilder:
    """
    Features acumulativas uptoW (semanas 0..W-1) SOLO dinámicas, normalizadas por curso-convocatoria.

    ✅ Para clustering accionable:
      - ELIMINADO: total_raw_clicks (sesga por volumen)
      - ELIMINADAS redundantes: density_uptoW, weekly_mean_uptoW, coeff_var_uptoW
      - rel_eng_zscore = proxy engagement (total_weighted_engagement), normalizada por curso.

    ✅ Añadidas (recency/momentum/concentración):
      - weeks_since_last_activity
      - streak_final
      - last_week_clicks_weighted
      - momentum_2w
      - top1_share
      - weeks_since_last_submission
    """

    features_root_dir: Path
    normalize_mode: str = "zscore"
    p_clip: float = 0.995
    use_log1p_counts: bool = True

    # column names
    inter_day_col: str = "date"
    inter_clicks_col: str = "sum_click"      # fallback "clicks"
    inter_activity_col: str = "activity_type"

    asm_deadline_col: str = "date"
    asm_submitted_col: str = "date_submitted"
    asm_score_col: str = "score"

    pass_threshold: float = 40.0  # para pass_ratio

    # Lista fija de activity_types para columnas clicks_*
    activity_types: Optional[list[str]] = None

    # Pesos por activity_type para total_weighted_engagement
    activity_weights: Optional[dict[str, float]] = None

    def __post_init__(self):
        self.features_root_dir = Path(self.features_root_dir)
        if self.normalize_mode not in {"zscore", "ratio_to_mean"}:
            raise ValueError("normalize_mode debe ser 'zscore' o 'ratio_to_mean'.")
        self.course_stats_: dict[int, dict] = {}

        if self.activity_types is None:
            self.activity_types = [
                "dataplus", "dualpane", "externalquiz", "folder", "forumng",
                "glossary", "homepage", "htmlactivity", "oucollaborate",
                "oucontent", "ouelluminate", "ouwiki", "page", "questionnaire",
                "quiz", "resource", "subpage", "url",
            ]

        if self.activity_weights is None:
            w = {a: 1.0 for a in self.activity_types}
            for a in ["quiz", "externalquiz", "questionnaire", "dataplus", "oucollaborate"]:
                if a in w:
                    w[a] = 1.3
            for a in ["homepage", "page", "subpage", "url", "folder", "resource"]:
                if a in w:
                    w[a] = 0.8
            self.activity_weights = w

    # ---------------- ids / course ----------------
    @staticmethod
    def _ensure_unique_id(df: pd.DataFrame) -> pd.DataFrame:
        if "unique_id" in df.columns:
            return df
        needed = {"id_student", "code_module", "code_presentation"}
        if not needed.issubset(df.columns):
            raise ValueError(f"Falta unique_id y no puedo construirlo (faltan {needed}).")
        out = df.copy()
        out["unique_id"] = (
            out["id_student"].astype(str)
            + "_"
            + out["code_module"].astype(str)
            + "_"
            + out["code_presentation"].astype(str)
        )
        return out

    @staticmethod
    def _course_series(df_students: pd.DataFrame, index_uids: pd.Index) -> pd.Series:
        s = AEUptoWFeaturesBuilder._ensure_unique_id(df_students.copy())
        s = s.drop_duplicates("unique_id").set_index("unique_id")
        if "code_module" in s.columns and "code_presentation" in s.columns:
            course = s["code_module"].astype(str) + "_" + s["code_presentation"].astype(str)
        else:
            course = pd.Series("UNKNOWN", index=s.index)
        return course.reindex(index_uids).fillna("UNKNOWN").astype(str)

    # ---------------- robust transforms ----------------
    def _clip_log1p(self, s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.clip(x, 0.0, None)
        if x.size:
            hi = float(np.quantile(x, self.p_clip))
            if not np.isfinite(hi) or hi < 0:
                hi = float(np.max(x))
            x = np.clip(x, 0.0, hi)
        if self.use_log1p_counts:
            x = np.log1p(x)
        return pd.Series(x, index=s.index)

    def _winsorize(self, s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        if not x.size:
            return pd.Series(x, index=s.index)
        lo = float(np.quantile(x, 1.0 - self.p_clip))
        hi = float(np.quantile(x, self.p_clip))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            return pd.Series(x, index=s.index)
        return pd.Series(np.clip(x, lo, hi), index=s.index)

    # ---------------- course normalization ----------------
    def _fit_course_stats(self, df: pd.DataFrame, course: pd.Series, cols: list[str]) -> dict:
        stats = {}
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
        if not isinstance(block_stats, dict):
            return 0.0, 1.0
        v = block_stats.get(col, None)
        if isinstance(v, dict):
            mu = float(v.get("mean", 0.0))
            sd = float(v.get("std", 1.0))
        elif v is None:
            mu, sd = 0.0, 1.0
        else:
            mu, sd = float(v), 1.0
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            sd = 1.0
        return mu, sd

    def _apply_course_norm(self, df: pd.DataFrame, course: pd.Series, stats: dict, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
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
            else:
                out.loc[idxs, cols] = block / (mu + EPS)

        out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def load_stats(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parsed = {}
        for k, v in raw.items():
            try:
                parsed[int(k)] = v
            except Exception:
                parsed[k] = v
        self.course_stats_ = parsed

    # ---------------- helpers for aggregates ----------------
    @staticmethod
    def _temporal_entropy(arr: np.ndarray) -> float:
        s = float(arr.sum())
        if s <= 0:
            return 0.0
        p = arr / (s + EPS)
        h = float(-(p * np.log(p + EPS)).sum())
        k = arr.size
        return float(h / (np.log(k + EPS)))

    @staticmethod
    def _max_streak(active_flags: np.ndarray) -> int:
        run = 0
        best = 0
        for v in active_flags:
            if v:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return int(best)

    @staticmethod
    def _first_active_week(active_flags: np.ndarray) -> int:
        idx = np.where(active_flags > 0)[0]
        return int(idx[0]) if idx.size else -1

    @staticmethod
    def _slope_from_weekly(arr_weekly: np.ndarray) -> float:
        n = arr_weekly.size
        if n <= 1:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom <= 0:
            return 0.0
        y = arr_weekly.astype(float)
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    def _weekly_clicks_by_type(
        self,
        df_interactions: pd.DataFrame,
        index_uids: pd.Index,
        maxW: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        inter = self._ensure_unique_id(df_interactions.copy())

        if self.inter_day_col not in inter.columns:
            raise ValueError(f"[interactions] falta '{self.inter_day_col}'")
        if self.inter_activity_col not in inter.columns:
            raise ValueError(f"[interactions] falta '{self.inter_activity_col}'")

        if self.inter_clicks_col in inter.columns:
            clicks = pd.to_numeric(inter[self.inter_clicks_col], errors="coerce").fillna(0.0).astype(float)
        elif "clicks" in inter.columns:
            clicks = pd.to_numeric(inter["clicks"], errors="coerce").fillna(0.0).astype(float)
        else:
            raise ValueError(f"[interactions] falta '{self.inter_clicks_col}' y 'clicks'")

        day = pd.to_numeric(inter[self.inter_day_col], errors="coerce").fillna(0.0).astype(float)
        week = np.floor(day / 7.0).astype(int)
        act = inter[self.inter_activity_col].astype(str).str.lower()

        inter = inter.assign(_clicks=clicks, _day=day, _week=week, _act=act)

        # prestart
        pre = inter[inter["_day"] < 0]
        prestart_clicks = pre.groupby("unique_id")["_clicks"].sum()
        prestart_clicks = prestart_clicks.reindex(index_uids).fillna(0.0)

        # weeks >=0
        cur = inter[(inter["_week"] >= 0) & (inter["_week"] < maxW)].copy()
        cur = cur[cur["_act"].isin(set(self.activity_types))].copy()

        cols = []
        for t in self.activity_types:
            for w in range(maxW):
                cols.append(f"clicks_{t}_w{w:02d}")

        if cur.empty:
            weekly_type = pd.DataFrame(0.0, index=index_uids, columns=cols)
            weekly_type.index.name = "unique_id"
            return weekly_type, prestart_clicks

        g = cur.groupby(["unique_id", "_act", "_week"])["_clicks"].sum()
        m = g.unstack(fill_value=0.0)  # index (uid, act) x week
        for w in range(maxW):
            if w not in m.columns:
                m[w] = 0.0
        m = m[sorted(m.columns)]

        frames = []
        acts_present = set(m.index.get_level_values(1)) if isinstance(m.index, pd.MultiIndex) else set()
        for t in self.activity_types:
            if t in acts_present:
                block = m.xs(t, level=1, drop_level=True)
            else:
                block = pd.DataFrame(0.0, index=[], columns=m.columns)

            block = block.reindex(index_uids, fill_value=0.0)
            block.columns = [f"clicks_{t}_w{w:02d}" for w in range(maxW)]
            frames.append(block)

        weekly_type = pd.concat(frames, axis=1).fillna(0.0)
        weekly_type = weekly_type.reindex(columns=cols, fill_value=0.0)
        weekly_type.index.name = "unique_id"
        return weekly_type, prestart_clicks

    def _weekly_total_clicks(self, weekly_type: pd.DataFrame, maxW: int) -> np.ndarray:
        arr = np.zeros((weekly_type.shape[0], maxW), dtype=float)
        for w in range(maxW):
            cols_w = [c for c in weekly_type.columns if c.endswith(f"_w{w:02d}")]
            if cols_w:
                arr[:, w] = weekly_type[cols_w].sum(axis=1).to_numpy(dtype=float)
        return arr

    # ---------------- assessments uptoW ----------------
    def _assessments_uptoW(self, df_assessments: pd.DataFrame, index_uids: pd.Index, W: int) -> pd.DataFrame:
        asm = self._ensure_unique_id(df_assessments.copy())

        out = pd.DataFrame(
            {
                "avg_score": np.zeros(len(index_uids), dtype=float),
                "score_std": np.zeros(len(index_uids), dtype=float),
                "late_ratio": np.zeros(len(index_uids), dtype=float),
                "pass_ratio": np.zeros(len(index_uids), dtype=float),
                "submission_count": np.zeros(len(index_uids), dtype=float),
                "score_slope": np.zeros(len(index_uids), dtype=float),
                "has_submitted": np.zeros(len(index_uids), dtype=float),
                # ✅ NUEVO
                "weeks_since_last_submission": np.full(len(index_uids), float(W), dtype=float),
            },
            index=index_uids,
        )

        if self.asm_submitted_col not in asm.columns:
            return out

        sub = pd.to_numeric(asm[self.asm_submitted_col], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).fillna(-9999.0).astype(float)
        week = np.floor(sub / 7.0).astype(int)

        asm = asm.assign(_week=week)
        asm = asm[(asm["_week"] >= 0) & (asm["_week"] < W)].copy()
        if asm.empty:
            return out

        # ✅ NUEVO: recency de submissions
        last_sub_week = asm.groupby("unique_id")["_week"].max().reindex(index_uids)
        last_sub_week = last_sub_week.fillna(-1).astype(int)
        wsls = np.where(last_sub_week >= 0, (W - 1) - last_sub_week.to_numpy(), float(W)).astype(float)
        out["weeks_since_last_submission"] = wsls

        score = pd.to_numeric(asm.get(self.asm_score_col, 0.0), errors="coerce").fillna(0.0).astype(float)
        deadline = pd.to_numeric(asm.get(self.asm_deadline_col, np.nan), errors="coerce").astype(float)

        # late: date_submitted > deadline
        sub_sel = pd.to_numeric(asm[self.asm_submitted_col], errors="coerce").fillna(0.0).astype(float)
        late = (sub_sel - deadline.loc[asm.index]).astype(float)
        late = late.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        asm = asm.assign(_score=score, _late=(late > 0).astype(int))

        g = asm.groupby("unique_id")
        out["submission_count"] = g.size().reindex(index_uids).fillna(0.0).astype(float)
        out["has_submitted"] = (out["submission_count"] > 0).astype(float)

        out["avg_score"] = g["_score"].mean().reindex(index_uids).fillna(0.0).astype(float)
        out["score_std"] = g["_score"].std().reindex(index_uids).fillna(0.0).astype(float)

        out["late_ratio"] = g["_late"].mean().reindex(index_uids).fillna(0.0).astype(float)
        out["pass_ratio"] = g["_score"].apply(
            lambda s: float((s >= self.pass_threshold).mean()) if len(s) else 0.0
        ).reindex(index_uids).fillna(0.0)

        # slope de score medio por semana
        by_week = asm.groupby(["unique_id", "_week"])["_score"].mean().unstack(fill_value=0.0)
        for w in range(W):
            if w not in by_week.columns:
                by_week[w] = 0.0
        by_week = by_week[sorted(by_week.columns)]
        by_week = by_week.reindex(index_uids, fill_value=0.0)

        slopes = []
        for uid in index_uids:
            slopes.append(self._slope_from_weekly(by_week.loc[uid].to_numpy(dtype=float)))
        out["score_slope"] = np.array(slopes, dtype=float)

        return out

    # ---------------- main: build per W ----------------
    def build_for_split(
        self,
        split: str,
        df_students: pd.DataFrame,
        df_interactions: pd.DataFrame,
        df_assessments: pd.DataFrame,
        windows: Iterable[int],
        *,
        fit: bool = False,
        save_stats: bool = True,
        min_weeks: int = 1,
    ) -> dict[int, Path]:
        windows = sorted({int(w) for w in windows if int(w) >= int(min_weeks)})
        if not windows:
            raise ValueError(f"windows vacío tras filtrar min_weeks={min_weeks}")
        maxW = max(windows)

        stu = self._ensure_unique_id(df_students.copy())
        stu = stu.drop_duplicates("unique_id").set_index("unique_id")
        index_uids = stu.index
        course = self._course_series(df_students, index_uids)

        weekly_type, prestart_clicks = self._weekly_clicks_by_type(df_interactions, index_uids, maxW)
        total_weekly = self._weekly_total_clicks(weekly_type, maxW)

        out_dir = self.features_root_dir / split / "ae_uptow_features"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[int, Path] = {}

        for W in windows:
            W = int(W)
            feat = {}

            # clicks_<type> acumulados
            for t in self.activity_types:
                cols_t = [c for c in weekly_type.columns if c.startswith(f"clicks_{t}_")]
                cols_use = [c for c in cols_t if int(c.split("_w")[-1]) < W]
                feat[f"clicks_{t}"] = weekly_type[cols_use].sum(axis=1).astype(float) if cols_use else 0.0

            # agregadas de clicks
            w_slice = total_weekly[:, :W]
            total_uptoW = w_slice.sum(axis=1)
            active_flags = (w_slice > 0).astype(int)
            active_weeks = active_flags.sum(axis=1).astype(float)

            # ✅ NUEVO: weeks_since_last_activity
            last_active_idx = np.full(len(index_uids), -1, dtype=int)
            for i in range(len(index_uids)):
                idxs = np.where(active_flags[i] > 0)[0]
                last_active_idx[i] = int(idxs[-1]) if idxs.size else -1
            weeks_since_last_activity = np.where(
                last_active_idx >= 0,
                (W - 1) - last_active_idx,
                float(W),
            ).astype(float)
            feat["weeks_since_last_activity"] = pd.Series(weeks_since_last_activity, index=index_uids)

            # ✅ NUEVO: streak_final (racha activa que termina en W-1)
            streak_final = np.zeros(len(index_uids), dtype=float)
            for i in range(len(index_uids)):
                flags = active_flags[i]
                r = 0
                for j in range(W - 1, -1, -1):
                    if flags[j] > 0:
                        r += 1
                    else:
                        break
                streak_final[i] = float(r)
            feat["streak_final"] = pd.Series(streak_final, index=index_uids)

            # weighted engagement acumulado
            weighted = np.zeros(len(index_uids), dtype=float)
            for t in self.activity_types:
                wgt = float(self.activity_weights.get(t, 1.0))
                weighted += wgt * pd.to_numeric(feat[f"clicks_{t}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            feat["total_weighted_engagement"] = pd.Series(weighted, index=index_uids)

            # ✅ NUEVO: last_week_clicks_weighted (semana W-1)
            if W >= 1:
                last_week_weighted = np.zeros(len(index_uids), dtype=float)
                for t in self.activity_types:
                    wgt = float(self.activity_weights.get(t, 1.0))
                    col = f"clicks_{t}_w{W-1:02d}"
                    if col in weekly_type.columns:
                        last_week_weighted += wgt * weekly_type[col].to_numpy(dtype=float)
                feat["last_week_clicks_weighted"] = pd.Series(last_week_weighted, index=index_uids)
            else:
                feat["last_week_clicks_weighted"] = pd.Series(0.0, index=index_uids)

            # ✅ NUEVO: momentum_2w (últimas 2 vs 2 previas)
            if W >= 4:
                last2 = w_slice[:, W-2:W].sum(axis=1)
                prev2 = w_slice[:, W-4:W-2].sum(axis=1)
                mom2 = (last2 - prev2) / (prev2 + EPS)
            elif W >= 2:
                last2 = w_slice[:, max(0, W-2):W].sum(axis=1)
                prev2 = np.zeros_like(last2)
                mom2 = (last2 - prev2) / (prev2 + EPS)
            else:
                mom2 = np.zeros(len(index_uids), dtype=float)
            feat["momentum_2w"] = pd.Series(mom2.astype(float), index=index_uids)

            # diversity (nº de tipos con clicks > 0)
            div = np.zeros(len(index_uids), dtype=float)
            for t in self.activity_types:
                div += (pd.to_numeric(feat[f"clicks_{t}"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0).astype(float)
            feat["activity_diversity"] = pd.Series(div, index=index_uids)

            # ✅ NUEVO: top1_share
            clicks_matrix = np.vstack([
                pd.to_numeric(feat[f"clicks_{t}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                for t in self.activity_types
            ]).T  # (n, n_types)
            top1 = clicks_matrix.max(axis=1)
            sum_types = clicks_matrix.sum(axis=1)
            feat["top1_share"] = pd.Series(top1 / (sum_types + EPS), index=index_uids)

            # slope / active weeks + (agregadas no redundantes)
            effort_slopes = []
            first_active = []
            streak_max = []
            entropy = []
            weekly_std = []
            active_ratio = []

            for i in range(len(index_uids)):
                arr = w_slice[i].astype(float)
                flags = active_flags[i]

                effort_slopes.append(self._slope_from_weekly(arr))
                first_active.append(self._first_active_week(flags))
                streak_max.append(self._max_streak(flags))
                entropy.append(self._temporal_entropy(arr))

                weekly_std.append(float(arr.std()))
                active_ratio.append(float(flags.sum() / (W + EPS)))

            feat["effort_slope"] = pd.Series(np.array(effort_slopes, dtype=float), index=index_uids)
            feat["active_weeks"] = pd.Series(active_weeks, index=index_uids)

            feat["weekly_std_uptoW"] = pd.Series(np.array(weekly_std, dtype=float), index=index_uids)
            feat["active_ratio_uptoW"] = pd.Series(np.array(active_ratio, dtype=float), index=index_uids)
            feat["first_active_week"] = pd.Series(np.array(first_active, dtype=float), index=index_uids)
            feat["streak_max_uptoW"] = pd.Series(np.array(streak_max, dtype=float), index=index_uids)
            feat["temporal_entropy_uptoW"] = pd.Series(np.array(entropy, dtype=float), index=index_uids)

            # prestart_ratio
            pre = prestart_clicks.to_numpy(dtype=float)
            feat["prestart_ratio"] = pd.Series(pre / (pre + total_uptoW + EPS), index=index_uids)

            # early_weeks_ratio (primeras 4)
            early_end = min(4, W)
            early_sum = w_slice[:, :early_end].sum(axis=1) if early_end > 0 else np.zeros_like(total_uptoW)
            feat["early_weeks_ratio"] = pd.Series(early_sum / (total_uptoW + EPS), index=index_uids)

            # curiosity_index
            feat["curiosity_index"] = pd.Series(div / (active_weeks + EPS), index=index_uids)

            # assessments (incluye weeks_since_last_submission)
            asm_feat = self._assessments_uptoW(df_assessments, index_uids, W)
            for c in asm_feat.columns:
                feat[c] = asm_feat[c]

            # api_index
            feat["api_index"] = pd.Series(
                pd.to_numeric(asm_feat["avg_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                * np.log1p(pd.to_numeric(asm_feat["submission_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)),
                index=index_uids,
            )

            dfW = pd.DataFrame(feat, index=index_uids).fillna(0.0)

            # transforms robustos (solo "count-like")
            count_cols = [
                c for c in dfW.columns
                if c.startswith("clicks_") or c in ["total_weighted_engagement", "last_week_clicks_weighted", "submission_count"]
            ]
            for c in count_cols:
                dfW[c] = self._clip_log1p(dfW[c])

            # winsorize en slopes
            for c in ["effort_slope", "score_slope", "momentum_2w"]:
                if c in dfW.columns:
                    dfW[c] = self._winsorize(dfW[c])

            # rel_eng_zscore proxy (se normaliza por curso)
            dfW["rel_eng_zscore"] = dfW["total_weighted_engagement"].astype(float)

            # normalización por curso-convocatoria
            norm_cols = list(dfW.columns)
            dfW[norm_cols] = dfW[norm_cols].astype(float)

            if fit:
                self.course_stats_[W] = self._fit_course_stats(dfW, course, cols=norm_cols)

            statsW = self.course_stats_.get(W)
            if statsW is not None:
                dfW = self._apply_course_norm(dfW, course, statsW, cols=norm_cols)

            dfW.index.name = "unique_id"

            out_path = out_dir / f"ae_uptow_features_w{W:02d}.csv"
            dfW.to_csv(out_path)
            saved[W] = out_path

        if fit and save_stats:
            stats_path = self.features_root_dir / "training" / "ae_uptow_features" / "course_stats_ae_uptow.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.course_stats_, f, ensure_ascii=False)

        return saved
