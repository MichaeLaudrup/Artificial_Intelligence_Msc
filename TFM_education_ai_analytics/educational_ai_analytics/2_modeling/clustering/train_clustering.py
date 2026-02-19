import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from educational_ai_analytics.config import (
    CLUSTERING_REPORTS_DIR,
    EMBEDDINGS_DATA_DIR,
    FEATURES_DATA_DIR,
    CLUSTERING_MODELS_DIR,
)

from .hyperparams import CLUSTERING_PARAMS

app = typer.Typer(add_completion=False)


def _parse_windows(windows: Optional[str]) -> List[int]:
    if not windows:
        return [int(w) for w in CLUSTERING_PARAMS.windows]
    return sorted({int(x.strip()) for x in windows.split(",") if x.strip()})


def _latent_path(split: str, window: int, latent_filename: str) -> Path:
    return EMBEDDINGS_DATA_DIR / split / f"upto_w{int(window):02d}" / latent_filename


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_gmm(X: np.ndarray, n_components: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type=CLUSTERING_PARAMS.covariance_type,
        reg_covar=CLUSTERING_PARAMS.reg_covar,
        n_init=CLUSTERING_PARAMS.n_init,
        max_iter=CLUSTERING_PARAMS.max_iter,
        init_params=CLUSTERING_PARAMS.init_params,
        random_state=CLUSTERING_PARAMS.random_state,
    )


def _k_diagnostics(X_std: np.ndarray, k_range: List[int]) -> pd.DataFrame:
    rows = []
    prev_bic = None
    prev_aic = None

    for k in k_range:
        gmm = _build_gmm(X_std, n_components=int(k))
        gmm.fit(X_std)

        bic = float(gmm.bic(X_std))
        aic = float(gmm.aic(X_std))

        bic_gain = (prev_bic - bic) if prev_bic is not None else np.nan
        aic_gain = (prev_aic - aic) if prev_aic is not None else np.nan

        bic_gain_pct = (bic_gain / abs(prev_bic) * 100.0) if prev_bic not in (None, 0) else np.nan
        aic_gain_pct = (aic_gain / abs(prev_aic) * 100.0) if prev_aic not in (None, 0) else np.nan

        rows.append(
            {
                "k": int(k),
                "bic": bic,
                "aic": aic,
                "delta_bic": bic_gain,
                "delta_aic": aic_gain,
                "delta_bic_pct": bic_gain_pct,
                "delta_aic_pct": aic_gain_pct,
            }
        )
        prev_bic = bic
        prev_aic = aic

    return pd.DataFrame(rows)


def _pick_k(diag: pd.DataFrame) -> Dict[str, Any]:
    if CLUSTERING_PARAMS.use_fixed_k:
        return {"k": int(CLUSTERING_PARAMS.n_clusters), "reason": "fixed_k"}

    d = diag.sort_values("k").copy()
    thr = float(CLUSTERING_PARAMS.marginal_gain_pct_threshold)
    cond = (d["delta_bic_pct"] < thr) & (d["delta_aic_pct"] < thr)

    if cond.any():
        return {
            "k": int(d.loc[cond, "k"].iloc[0]),
            "reason": f"first_k_with_marginal_gain_below_{thr:.2f}pct",
        }

    if CLUSTERING_PARAMS.use_bic:
        return {"k": int(d.loc[d["bic"].idxmin(), "k"]), "reason": "min_bic"}

    if CLUSTERING_PARAMS.use_aic:
        return {"k": int(d.loc[d["aic"].idxmin(), "k"]), "reason": "min_aic"}

    return {"k": int(CLUSTERING_PARAMS.n_clusters), "reason": "fallback_fixed_k"}


def _cluster_stats_from_target(labels: np.ndarray, df_target: pd.DataFrame, k: int) -> pd.DataFrame:
    if "final_result" not in df_target.columns:
        raise ValueError("target.csv must contain 'final_result' column")

    y = df_target["final_result"].astype(int).values
    is_withdrawn = (y == 0).astype(int)
    is_success = np.isin(y, [2, 3]).astype(int)

    tmp = pd.DataFrame(
        {
            "cluster_id": labels,
            "is_withdrawn": is_withdrawn,
            "is_success": is_success,
        },
        index=df_target.index,
    )

    grp = (
        tmp.groupby("cluster_id")
        .agg(n=("cluster_id", "size"), withdrawn_rate=("is_withdrawn", "mean"), success_rate=("is_success", "mean"))
        .reset_index()
    )

    present = set(grp["cluster_id"].tolist())
    missing = [c for c in range(k) if c not in present]
    if missing:
        filler = pd.DataFrame(
            {
                "cluster_id": missing,
                "n": [0] * len(missing),
                "withdrawn_rate": [0.0] * len(missing),
                "success_rate": [0.0] * len(missing),
            }
        )
        grp = pd.concat([grp, filler], ignore_index=True)

    grp["withdrawn_pct"] = 100.0 * grp["withdrawn_rate"]
    grp["success_pct"] = 100.0 * grp["success_rate"]
    return grp.sort_values("cluster_id").reset_index(drop=True)


def _build_mapping_from_target(stats: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    worst_to_best = stats.sort_values(["withdrawn_rate", "success_rate"], ascending=[False, True]).reset_index(drop=True)
    k = len(stats)

    if k == 5:
        label_pack = [
            ("CRITICAL_RISK_INACTIVE", "Riesgo crítico (inactivos)"),
            ("STANDARD_PROFILE", "Perfil estándar"),
            ("METHODICAL_EXPLORER", "Exploradores metódicos"),
            ("CONSISTENT_GOOD", "Consistentes (buen nivel)"),
            ("STRATEGIC_HIGH_PERFORMER", "Alto rendimiento (estratégico)"),
        ]
    elif k == 6:
        label_pack = [
            ("CRITICAL_RISK_INACTIVE", "Riesgo crítico (inactivos)"),
            ("STANDARD_PROFILE", "Perfil estándar"),
            ("METHODICAL_EXPLORER", "Exploradores metódicos"),
            ("CONSISTENT_GOOD", "Consistentes (buen nivel)"),
            ("ENGAGED_FATIGUE", "Comprometidos (con fatiga)"),
            ("STRATEGIC_HIGH_PERFORMER", "Alto rendimiento (estratégico)"),
        ]
    else:
        label_pack = [(f"CLUSTER_{i}", f"Cluster {i}") for i in range(k)]

    mapping: Dict[str, Dict[str, Any]] = {}
    for rank, row in worst_to_best.iterrows():
        cid = int(row["cluster_id"])
        label, name = label_pack[min(rank, len(label_pack) - 1)]
        mapping[str(cid)] = {
            "label": label,
            "name": name,
            "rank_worst_to_best": int(rank),
            "n": int(row["n"]),
            "withdrawn_pct": float(row["withdrawn_pct"]),
            "success_pct": float(row["success_pct"]),
        }
    return mapping


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    sample_n = min(int(CLUSTERING_PARAMS.silhouette_sample_size), max(2, n - 1))
    if sample_n < 2:
        return float("nan")
    return float(
        silhouette_score(
            X,
            labels,
            sample_size=sample_n,
            random_state=CLUSTERING_PARAMS.random_state,
        )
    )


def _compute_metrics(X_std: np.ndarray, gmm: GaussianMixture, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    counts = np.bincount(labels, minlength=int(gmm.n_components))
    confidence = probs.max(axis=1)

    metrics = {
        "n_students": int(X_std.shape[0]),
        "latent_dim": int(X_std.shape[1]),
        "k": int(gmm.n_components),
        "bic": float(gmm.bic(X_std)),
        "aic": float(gmm.aic(X_std)),
        "log_likelihood_mean": float(gmm.score(X_std)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_median": float(np.median(confidence)),
        "doubtful_threshold": float(CLUSTERING_PARAMS.confidence_doubtful_threshold),
        "doubtful_pct": float(np.mean(confidence < CLUSTERING_PARAMS.confidence_doubtful_threshold)),
        "cluster_sizes": {str(i): int(counts[i]) for i in range(len(counts))},
        "min_cluster_pct": float(counts.min() / counts.sum()) if counts.sum() else 0.0,
        "silhouette": _safe_silhouette(X_std, labels),
        "calinski_harabasz": float(calinski_harabasz_score(X_std, labels)) if len(np.unique(labels)) > 1 else float("nan"),
        "davies_bouldin": float(davies_bouldin_score(X_std, labels)) if len(np.unique(labels)) > 1 else float("nan"),
    }
    return metrics


@app.command()
def main(
    windows: Optional[str] = typer.Option(
        None,
        help="Ventanas separadas por coma (ej: 12,18,24). Si se omite usa W_WINDOWS de config.py.",
    ),
    latent_filename: str = typer.Option(CLUSTERING_PARAMS.latent_filename, help="Nombre del CSV latente."),
    split_train: str = typer.Option(CLUSTERING_PARAMS.split_train, help="Split usado para entrenar."),
    save_legacy_latest: bool = typer.Option(
        True,
        help="Además de artefactos por ventana, guarda alias legacy para la última ventana.",
    ),
):
    """
    Entrena GMM en producción para cada ventana W en W_WINDOWS.

    Salidas por ventana (models/clustering_models/):
      - gmm_ae_wXX.joblib
      - scaler_latent_ae_wXX.joblib
      - cluster_mapping_wXX.json

    Reportes (reports/clustering/):
      - train_clustering_metrics_wXX.json
      - gmm_k_diagnostics.csv
    """
    selected_windows = _parse_windows(windows)
    logger.info(f"🧠 Clustering TRAIN | split={split_train} | windows={selected_windows}")

    CLUSTERING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_diag_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for w in selected_windows:
        latent_path = _latent_path(split=split_train, window=w, latent_filename=latent_filename)
        df_lat = _safe_read_csv(latent_path)
        if df_lat.empty:
            logger.warning(f"⚠️ W{w:02d}: embeddings vacíos/no encontrados en {latent_path}")
            continue

        X = df_lat.values.astype(np.float32)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        if CLUSTERING_PARAMS.use_fixed_k:
            diag = pd.DataFrame()
            k_pick = {"k": int(CLUSTERING_PARAMS.n_clusters), "reason": "fixed_k"}
        else:
            diag = _k_diagnostics(X_std, k_range=[int(k) for k in CLUSTERING_PARAMS.gmm_k_range])
            diag["window"] = int(w)
            all_diag_rows.extend(diag.to_dict(orient="records"))
            k_pick = _pick_k(diag)

        k_final = int(k_pick["k"])

        gmm = _build_gmm(X_std, n_components=k_final)
        gmm.fit(X_std)

        probs = gmm.predict_proba(X_std)
        labels = probs.argmax(axis=1).astype(int)

        target_path = FEATURES_DATA_DIR / split_train / "target.csv"
        df_target = _safe_read_csv(target_path)
        common = df_lat.index.intersection(df_target.index)

        if len(common) > 0 and "final_result" in df_target.columns:
            idx_pos = df_lat.index.get_indexer(common)
            stats = _cluster_stats_from_target(labels[idx_pos], df_target.loc[common], k=k_final)
            mapping = _build_mapping_from_target(stats)
            stats_records = stats.to_dict(orient="records")
        else:
            mapping = {str(i): {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"} for i in range(k_final)}
            stats_records = []

        metrics = _compute_metrics(X_std, gmm, probs, labels)
        metrics["window"] = int(w)
        metrics["latent_path"] = str(latent_path)
        metrics["k_selection_reason"] = str(k_pick["reason"])
        metrics["cluster_outcome_stats_train"] = stats_records

        out_gmm = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}_w{w:02d}.joblib"
        out_scaler = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}_w{w:02d}.joblib"
        out_mapping = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}_w{w:02d}.json"
        out_metrics = CLUSTERING_REPORTS_DIR / f"{CLUSTERING_PARAMS.metrics_prefix}_w{w:02d}.json"

        joblib.dump(gmm, out_gmm)
        joblib.dump(scaler, out_scaler)
        out_mapping.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
        out_metrics.write_text(
            json.dumps(
                {
                    "hyperparams": asdict(CLUSTERING_PARAMS),
                    "selected_k": k_final,
                    "k_selection": k_pick,
                    "k_diagnostics": diag.to_dict(orient="records") if not diag.empty else [],
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        summary_rows.append(
            {
                "window": int(w),
                "k": int(k_final),
                "bic": float(metrics["bic"]),
                "aic": float(metrics["aic"]),
                "silhouette": float(metrics["silhouette"]),
                "min_cluster_pct": float(metrics["min_cluster_pct"]),
            }
        )

        logger.success(
            f"✅ W{w:02d} | K={k_final} ({k_pick['reason']}) | "
            f"BIC={metrics['bic']:.1f} | Sil={metrics['silhouette']:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("No se entrenó ningún modelo GMM: revisa embeddings y ventanas.")

    diag_path = CLUSTERING_REPORTS_DIR / CLUSTERING_PARAMS.diagnostics_filename
    if all_diag_rows:
        pd.DataFrame(all_diag_rows).to_csv(diag_path, index=False)
    elif diag_path.exists():
        diag_path.unlink()

    summary_path = CLUSTERING_REPORTS_DIR / "train_clustering_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))

    if save_legacy_latest:
        max_w = max([int(r["window"]) for r in summary_rows])
        src_gmm = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}_w{max_w:02d}.joblib"
        src_scaler = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}_w{max_w:02d}.joblib"
        src_mapping = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}_w{max_w:02d}.json"

        dst_gmm = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}.joblib"
        dst_scaler = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}.joblib"
        dst_mapping = CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}.json"

        joblib.dump(joblib.load(src_gmm), dst_gmm)
        joblib.dump(joblib.load(src_scaler), dst_scaler)
        dst_mapping.write_text(src_mapping.read_text())

        logger.info(f"📌 Alias legacy actualizados con W{max_w:02d} en {CLUSTERING_MODELS_DIR}")

    logger.success(f"📄 Diagnóstico K guardado en: {diag_path}")
    logger.success(f"📄 Resumen TRAIN guardado en: {summary_path}")


if __name__ == "__main__":
    app()
