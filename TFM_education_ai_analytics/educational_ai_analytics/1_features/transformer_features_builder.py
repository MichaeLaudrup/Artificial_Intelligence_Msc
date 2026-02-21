import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransformerFeatureArtifacts:
    upto_week: int
    X_seq: np.ndarray          # (N, T, F)
    mask: np.ndarray           # (N, T)
    y: np.ndarray              # (N,)
    ids: np.ndarray            # (N,) strings
    X_static: Optional[np.ndarray] = None  # (N, S) si aplica


class TransformerFeaturesBuilder:
    """
    Construye features tipo Transformer:
      - Secuencias: clicks agregados por (week, activity_type) hasta W (exclusive)
      - Máscara temporal por semana (si hay actividad esa semana)
      - Labels (target)
      - Features estáticas de segmentación (prob. clusters + entropy) alineadas a ids
    """

    def __init__(
        self,
        out_root_dir: Path,
        segmented_root_dir: Path,
        features_root_dir: Path,
        *,
        windows: List[int],
        features_cluster: Optional[List[str]] = None,
        inter_day_col: str = "date",
        inter_clicks_cols: Tuple[str, ...] = ("sum_click", "clicks"),
        inter_activity_col: str = "activity_type",
        students_unreg_col: str = "date_unregistration",
        students_id_cols: Tuple[str, str, str] = ("id_student", "code_module", "code_presentation"),
        target_file_name: str = "target.csv",
        target_col: str = "final_result",
        meta_file: str = "transformer_meta.json",
    ):
        self.out_root_dir = Path(out_root_dir)
        self.segmented_root_dir = Path(segmented_root_dir)
        self.features_root_dir = Path(features_root_dir)
        self.windows = sorted(list(windows))

        self.features_cluster = features_cluster or [
            "p_cluster_0", "p_cluster_1", "p_cluster_2",
            "p_cluster_3", "p_cluster_4", "entropy_norm"
        ]

        self.inter_day_col = inter_day_col
        self.inter_clicks_cols = inter_clicks_cols
        self.inter_activity_col = inter_activity_col

        self.students_unreg_col = students_unreg_col
        self.students_id_cols = students_id_cols

        self.target_file_name = target_file_name
        self.target_col = target_col

        self.meta_path = self.out_root_dir / meta_file
        self.activities_global: Optional[List[str]] = None
        self.scalers: Dict[str, Dict[str, list]] = {}

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ensure_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if "unique_id" in df.columns:
            return df
        a, b, c = self.students_id_cols
        needed = {a, b, c}
        if not needed.issubset(df.columns):
            return df
        out = df.copy()
        out["unique_id"] = (
            out[a].astype(str) + "_" + out[b].astype(str) + "_" + out[c].astype(str)
        )
        return out

    def _pick_click_col(self, interactions_df: pd.DataFrame) -> str:
        for c in self.inter_clicks_cols:
            if c in interactions_df.columns:
                return c
        raise ValueError(
            f"No se encontró ninguna columna de clicks {self.inter_clicks_cols} en interactions.csv"
        )

    def _compute_week(self, s: pd.Series) -> pd.Series:
        # week 0-based a partir de días
        return (pd.to_numeric(s, errors="coerce").fillna(-9999) // 7).astype(int)

    def _load_target(self, split: str) -> pd.DataFrame:
        fp = self.features_root_dir / split / self.target_file_name
        if not fp.exists():
            raise FileNotFoundError(
                f"No existe target para split={split}: {fp}. "
                f"(Normalmente lo genera tu pipeline de 3_features.)"
            )
        tgt = pd.read_csv(fp, index_col=0).sort_index()
        if self.target_col not in tgt.columns:
            raise ValueError(f"Target sin columna '{self.target_col}': {fp}")
        return tgt

    def _save_meta(self):
        if self.activities_global is None:
            return
        payload = {
            "activities_global": self.activities_global,
            "features_cluster": self.features_cluster,
            "windows": self.windows,
            "scalers": self.scalers,
        }
        self.out_root_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_meta(self):
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"No existe meta en {self.meta_path}. Ejecuta primero training (fit=True)."
            )
        with open(self.meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.activities_global = payload["activities_global"]
        self.scalers = payload.get("scalers", {})

    # ----------------------------
    # Core builders
    # ----------------------------
    def fit_activities_global(self, interactions_df: pd.DataFrame):
        interactions_df = interactions_df.copy()
        interactions_df[self.inter_activity_col] = (
            interactions_df[self.inter_activity_col].astype(str).str.strip().str.lower()
        )
        interactions_df["week"] = self._compute_week(interactions_df[self.inter_day_col])
        base = interactions_df[interactions_df["week"] >= 0]
        self.activities_global = sorted(base[self.inter_activity_col].unique().tolist())
        logger.info(f"🧠 activities_global (F={len(self.activities_global)}): ok")
        self._save_meta()

    def build_for_split(
        self,
        *,
        split: str,
        students_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        fit: bool,
    ) -> Dict[int, Path]:
        """
        Genera y guarda por cada W:
          data/6_transformer_features/<split>/transformer_uptoW{W}.npz
        Devuelve dict W -> filepath
        """
        out_dir = self.out_root_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        students_df = self._ensure_unique_id(students_df)
        interactions_df = self._ensure_unique_id(interactions_df)

        # target (index = unique_id)
        target_full = self._load_target(split)

        # clicks col robusta
        click_col = self._pick_click_col(interactions_df)

        # limpieza activity_type + week
        interactions_df = interactions_df.copy()
        interactions_df[self.inter_activity_col] = (
            interactions_df[self.inter_activity_col].astype(str).str.strip().str.lower()
        )
        interactions_df["week"] = self._compute_week(interactions_df[self.inter_day_col])

        # base temporal válida
        interactions_base = interactions_df[interactions_df["week"] >= 0].copy()

        # activities global: fit en training, load en el resto
        if fit:
            self.fit_activities_global(interactions_base)
        else:
            if self.activities_global is None:
                self._load_meta()

        assert self.activities_global is not None
        activities_global = self.activities_global

        # semanas de abandono
        st = students_df.copy()
        st["week_unregistration"] = (
            pd.to_numeric(st.get(self.students_unreg_col, np.nan), errors="coerce")
            .fillna(9999) // 7
        ).astype(int)

        saved: Dict[int, Path] = {}

        for upto_week in self.windows:
            inter_uptoW = interactions_base[interactions_base["week"] < upto_week].copy()

            # Evitar alumnos que abandonaron antes de la semana objeto
            active_students = st[st["week_unregistration"] >= upto_week]
            valid_ids = active_students["unique_id"].unique()

            total_students = inter_uptoW["unique_id"].nunique()
            removed = total_students - len(valid_ids)
            logger.info(f"[{split}] W={upto_week}: eliminados {removed} por abandono temprano")
            inter_uptoW = inter_uptoW[inter_uptoW["unique_id"].isin(valid_ids)]

            # agregación
            g = (
                inter_uptoW.groupby(["unique_id", "week", self.inter_activity_col], as_index=False)[click_col]
                .sum()
                .rename(columns={click_col: "sum_click"})
            )

            # pivot wide con columnas completas (weeks x activities_global)
            weeks = list(range(upto_week))
            full_cols = pd.MultiIndex.from_product(
                [weeks, activities_global], names=["week", self.inter_activity_col]
            )

            wide = (
                g.pivot_table(
                    index="unique_id",
                    columns=["week", self.inter_activity_col],
                    values="sum_click",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(columns=full_cols, fill_value=0)
                .sort_index()
            )

            # orden estable basado en target (trazabilidad y eval)
            common_ids = target_full.index.intersection(wide.index)
            wide_w = wide.loc[common_ids]
            target_w = target_full.loc[common_ids]

            # secuencia (N, T, F)
            X_seq = wide_w.values.reshape(
                len(wide_w), upto_week, len(activities_global)
            ).astype(np.float32)

            # máscara semanal (N, T)
            mask = (X_seq.sum(axis=2) > 0).astype(np.int32)
            
            # --- NORMALIZACION Log1p + Z-Score (calculado solo en split=training) ---
            X_seq_log = np.log1p(X_seq)
            w_key = str(upto_week)
            if fit:
                active = mask.reshape(-1).astype(bool)
                X_seq_flat = X_seq_log.reshape(-1, len(activities_global))[active]
                mu = X_seq_flat.mean(axis=0)
                std = X_seq_flat.std(axis=0) + 1e-8
                self.scalers[w_key] = {"mu": mu.tolist(), "std": std.tolist()}
            else:
                if w_key not in self.scalers:
                    raise KeyError(f"Scaler no encontrado para W={upto_week}. Ejecuta training primero o revisa meta.")
                mu = np.asarray(self.scalers[w_key]["mu"], dtype=np.float32)
                std = np.asarray(self.scalers[w_key]["std"], dtype=np.float32)
                
            X_seq_norm = (X_seq_log - mu) / std
            X_seq = (X_seq_norm * np.expand_dims(mask, axis=-1)).astype(np.float32)

            # labels
            y = target_w[self.target_col].astype(np.int64).values

            # static segmented (opcional pero recomendado)
            X_static = self._load_segmented_static(split=split, upto_week=upto_week, uid=common_ids)

            # guardado .npz
            fp = out_dir / f"transformer_uptoW{upto_week}.npz"
            np.savez_compressed(
                fp,
                X_seq=X_seq,
                mask=mask,
                y=y,
                ids=common_ids.astype(str).values,
                X_static=X_static if X_static is not None else np.zeros((len(common_ids), 0), dtype=np.float32),
                activities=np.array(activities_global, dtype=object),
                features_cluster=np.array(self.features_cluster, dtype=object),
                upto_week=np.int32(upto_week),
            )
            saved[upto_week] = fp

            logger.info(
                f"[{split}] ✅ W={upto_week} | X_seq={X_seq.shape} mask={mask.shape} "
                f"y={y.shape} X_static={(None if X_static is None else X_static.shape)} -> {fp.name}"
            )

        if fit:
            # Re-guardar meta para incluir los scalers calculados en este split de training
            self._save_meta()

        return saved

    def _load_segmented_static(self, *, split: str, upto_week: int, uid: pd.Index) -> Optional[np.ndarray]:
        """
        Lee 5_students_segmented/<split>/students_segmented_uptoW{W}.csv
        y junta con variables demográficas de day0_static_features.csv
        """
        fp = self.segmented_root_dir / split / f"students_segmented_uptoW{upto_week}.csv"
        if not fp.exists():
            logger.warning(f"[{split}] ⚠️ No existe segmented file: {fp} (saltando X_static)")
            return None

        cols_to_load = ["unique_id"] + self.features_cluster
        seg = pd.read_csv(fp, usecols=lambda c: c in cols_to_load)
        if "unique_id" not in seg.columns:
            logger.warning(f"[{split}] ⚠️ Segmented sin unique_id: {fp} (saltando X_static)")
            return None

        seg = seg.set_index("unique_id")
        
        # Añadir demográficos pre-calculados (day0) para multiplicar la fuerza de la red estática
        day0_fp = self.features_root_dir / split / "day0_static_features.csv"
        if day0_fp.exists():
            day0_df = pd.read_csv(day0_fp).set_index("unique_id")
            combined_df = pd.concat([seg, day0_df], axis=1)
        else:
            logger.warning(f"[{split}] ⚠️ No existe {day0_fp}, usando sólo clusters.")
            combined_df = seg

        X_static_df = combined_df.reindex(uid).fillna(0.0)

        # garantizar orden de columnas original + demográficas ordenadas para consistencia
        cluster_cols = [c for c in self.features_cluster if c in X_static_df.columns]
        demo_cols = sorted([c for c in X_static_df.columns if c not in cluster_cols])
        
        final_cols = cluster_cols + demo_cols
        X_static_df = X_static_df[final_cols]

        return X_static_df.values.astype(np.float32)