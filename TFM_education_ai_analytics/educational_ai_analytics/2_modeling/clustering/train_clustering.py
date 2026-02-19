import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from educational_ai_analytics.config import (
    EMBEDDINGS_DATA_DIR,
    FEATURES_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    W_WINDOWS,
)

app = typer.Typer(add_completion=False)


# -------------------------
# Config (production-safe)
# -------------------------
@dataclass(frozen=True)
class ClusteringConfig:
    k: int = 6
    covariance_type: str = "diag"
    n_init: int = 5
    max_iter: int = 800
    reg_covar: float = 1e-6
    seed: int = 42
    doubtful_threshold: float = 0.60


# -------------------------
# Default paths
# -------------------------
DEFAULT_WINDOW = int(max(W_WINDOWS)) if W_WINDOWS else 24
EMBED_TRAIN_PATH = EMBEDDINGS_DATA_DIR / "training" / f"upto_w{DEFAULT_WINDOW:02d}" / "ae_latent.csv"
TARGET_TRAIN_PATH = FEATURES_DATA_DIR / "training" / "target.csv"
OUT_GMM = MODELS_DIR / "gmm_ae.joblib"
OUT_SCALER = MODELS_DIR / "scaler_latent_ae.joblib"
OUT_MAPPING = MODELS_DIR / "cluster_mapping.json"
OUT_METRICS = REPORTS_DIR / "train_clustering_metrics.json"


# -------------------------
# Helpers
# -------------------------
def load_latent_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.sort_index()


def resolve_embeddings_path(path: Optional[Path], window: int, latent_filename: str) -> Path:
    if path is not None:
        return path
    return EMBEDDINGS_DATA_DIR / "training" / f"upto_w{int(window):02d}" / latent_filename


def load_target(path: Path, index_like: pd.Index) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    df = pd.read_csv(path, index_col=0).sort_index()
    common = index_like.intersection(df.index)
    df = df.loc[common].copy()
    return df


def fit_scaler_and_gmm(
    df_latent: pd.DataFrame,
    cfg: ClusteringConfig
) -> tuple[StandardScaler, GaussianMixture, np.ndarray, np.ndarray]:
    """Returns scaler, gmm, probs_max (N,), labels (N,) on TRAIN."""
    X = df_latent.values.astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=cfg.k,
        covariance_type=cfg.covariance_type,
        random_state=cfg.seed,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        reg_covar=cfg.reg_covar,
        init_params="kmeans",
    )
    gmm.fit(Xs)

    probs = gmm.predict_proba(Xs)
    probs_max = probs.max(axis=1)
    labels = probs.argmax(axis=1).astype(int)
    return scaler, gmm, probs_max, labels


def build_default_mapping(k: int) -> Dict[str, Dict[str, str]]:
    return {str(i): {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"} for i in range(k)}


def _cluster_stats_from_target(
    labels: np.ndarray,
    df_target: pd.DataFrame,
    k: int
) -> pd.DataFrame:
    """
    df_target expects column 'final_result' with mapping:
      0 Withdrawn, 1 Fail, 2 Pass, 3 Distinction
    """
    if "final_result" not in df_target.columns:
        raise ValueError("target.csv must contain 'final_result' column")

    y = df_target["final_result"].astype(int).values
    is_withdrawn = (y == 0).astype(int)
    is_success = np.isin(y, [2, 3]).astype(int)

    tmp = pd.DataFrame({
        "cluster_id": labels,
        "is_withdrawn": is_withdrawn,
        "is_success": is_success,
    }, index=df_target.index)

    grp = tmp.groupby("cluster_id").agg(
        n=("cluster_id", "size"),
        withdrawn_rate=("is_withdrawn", "mean"),
        success_rate=("is_success", "mean"),
    ).reset_index()

    # asegurar todos los clusters
    present = set(grp["cluster_id"].tolist())
    missing = [c for c in range(k) if c not in present]
    if missing:
        filler = pd.DataFrame({
            "cluster_id": missing,
            "n": [0]*len(missing),
            "withdrawn_rate": [0.0]*len(missing),
            "success_rate": [0.0]*len(missing),
        })
        grp = pd.concat([grp, filler], ignore_index=True)

    grp["withdrawn_pct"] = 100.0 * grp["withdrawn_rate"]
    grp["success_pct"] = 100.0 * grp["success_rate"]
    return grp.sort_values("cluster_id").reset_index(drop=True)


def build_mapping_from_target(
    stats: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Asigna etiquetas/nombres de forma estable basándose en TRAIN:
      - Peor (max withdrawn) -> CRITICAL_RISK_INACTIVE
      - Mejor (max success)  -> STRATEGIC_HIGH_PERFORMER
      - Intermedios: STANDARD_PROFILE / METHODICAL_EXPLORER / CONSISTENT_GOOD / ENGAGED_FATIGUE
    """
    # ranking por withdrawn desc (peor arriba)
    worst_to_best = stats.sort_values(["withdrawn_rate", "success_rate"], ascending=[False, True]).reset_index(drop=True)
    best_to_worst = stats.sort_values(["success_rate", "withdrawn_rate"], ascending=[False, True]).reset_index(drop=True)

    k = len(stats)

    # etiquetas “human-friendly” (ajusta a tu gusto)
    # Orden conceptual: peor → mejor
    # etiquetas “human-friendly”
    # Orden conceptual: peor (worst) → mejor (best)
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

    # asignación por ranking worst->best
    mapping: Dict[str, Dict[str, Any]] = {}
    for rank, row in worst_to_best.iterrows():
        cid = int(row["cluster_id"])
        label, name = label_pack[min(rank, len(label_pack)-1)]
        mapping[str(cid)] = {
            "label": label,
            "name": name,
            "rank_worst_to_best": int(rank),
            "n": int(row["n"]),
            "withdrawn_pct": float(row["withdrawn_pct"]),
            "success_pct": float(row["success_pct"]),
        }

    # extra: guardar también quién es el “best” según success por claridad
    best_cid = int(best_to_worst.iloc[0]["cluster_id"])
    worst_cid = int(worst_to_best.iloc[0]["cluster_id"])
    mapping["_meta"] = {
        "best_by_success_cluster_id": best_cid,
        "worst_by_withdrawn_cluster_id": worst_cid,
        "note": "Mapping generado automáticamente usando target TRAIN (solo para nombrar, no afecta al clustering).",
    }
    return mapping


def compute_metrics(
    df_latent: pd.DataFrame,
    scaler: StandardScaler,
    gmm: GaussianMixture,
    probs_max: np.ndarray,
    labels: np.ndarray,
    cfg: ClusteringConfig,
) -> Dict[str, Any]:
    counts = np.bincount(labels, minlength=cfg.k)
    min_cluster_pct = float(counts.min() / counts.sum()) if counts.sum() else 0.0
    doubtful_pct = float(np.mean(probs_max < cfg.doubtful_threshold))

    X = df_latent.values.astype(np.float32)
    Xs = scaler.transform(X)

    metrics: Dict[str, Any] = {
        "n_students": int(len(df_latent)),
        "latent_dim": int(df_latent.shape[1]),
        "k": int(cfg.k),
        "covariance_type": cfg.covariance_type,
        "seed": int(cfg.seed),
        "n_init": int(cfg.n_init),
        "max_iter": int(cfg.max_iter),
        "reg_covar": float(cfg.reg_covar),
        "log_likelihood_mean": float(gmm.score(Xs)),
        "bic": float(gmm.bic(Xs)),
        "aic": float(gmm.aic(Xs)),
        "confidence_mean": float(np.mean(probs_max)),
        "confidence_median": float(np.median(probs_max)),
        "doubtful_threshold": float(cfg.doubtful_threshold),
        "doubtful_pct": float(doubtful_pct),
        "min_cluster_pct": float(min_cluster_pct),
        "cluster_sizes": {str(i): int(counts[i]) for i in range(cfg.k)},
    }
    return metrics


# -------------------------
# CLI
# -------------------------
@app.command()
def main(
    embeddings_path: Optional[Path] = typer.Option(
        None,
        help="Ruta manual al CSV latente de training. Si no se indica, usa embeddings_dir/training/upto_wXX/ae_latent.csv.",
    ),
    window: int = typer.Option(DEFAULT_WINDOW, help="Ventana W a usar (upto_wXX)."),
    latent_filename: str = typer.Option("ae_latent.csv", help="Nombre del CSV latente dentro de upto_wXX."),
    target_path: Path = typer.Option(TARGET_TRAIN_PATH, help="TRAIN target.csv (para auto-nombrar clusters)."),
    out_gmm: Path = typer.Option(OUT_GMM, help="Output path for fitted GMM artifact (.joblib)."),
    out_scaler: Path = typer.Option(OUT_SCALER, help="Output path for fitted scaler artifact (.joblib)."),
    out_mapping: Path = typer.Option(OUT_MAPPING, help="Output path for cluster mapping JSON."),
    out_metrics: Path = typer.Option(OUT_METRICS, help="Output path for training metrics JSON."),

    k: int = typer.Option(6, help="Number of GMM components."),
    covariance_type: str = typer.Option("diag", help="GMM covariance_type: 'full'|'tied'|'diag'|'spherical'."),
    n_init: int = typer.Option(5, help="GMM n_init."),
    max_iter: int = typer.Option(800, help="GMM max_iter."),
    reg_covar: float = typer.Option(1e-6, help="GMM reg_covar (stability)."),
    seed: int = typer.Option(42, help="Random seed (single seed in production)."),
    doubtful_threshold: float = typer.Option(0.60, help="Posterior max threshold below which assignments are 'doubtful'."),

    auto_mapping: bool = typer.Option(True, help="Si True, genera cluster_mapping.json automáticamente usando target TRAIN."),
    mapping_json_in: Optional[Path] = typer.Option(None, help="(Override) Mapping JSON manual a copiar."),
):
    """
    Production training for clustering (AE latent + GMM) + mapping dinámico:
    - Fit scaler + GMM en TRAIN
    - Guarda artefactos
    - Genera mapping human-friendly automáticamente (usando target TRAIN) para estabilidad entre ejecuciones
    """
    cfg = ClusteringConfig(
        k=k,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        seed=seed,
        doubtful_threshold=doubtful_threshold,
    )

    latent_path = resolve_embeddings_path(embeddings_path, window=window, latent_filename=latent_filename)
    df_latent = load_latent_embeddings(latent_path)
    scaler, gmm, probs_max, labels = fit_scaler_and_gmm(df_latent, cfg)
    metrics = compute_metrics(df_latent, scaler, gmm, probs_max, labels, cfg)
    metrics["window"] = int(window)
    metrics["embeddings_path"] = str(latent_path)

    # Ensure dirs
    out_gmm.parent.mkdir(parents=True, exist_ok=True)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    out_mapping.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    joblib.dump(gmm, out_gmm)
    joblib.dump(scaler, out_scaler)

    # ---- Mapping: prioridad -> mapping_json_in > auto_mapping(target) > default
    if mapping_json_in is not None:
        if not mapping_json_in.exists():
            raise FileNotFoundError(f"Provided mapping_json_in does not exist: {mapping_json_in}")
        mapping = json.loads(mapping_json_in.read_text())

    elif auto_mapping:
        # genera mapping basado en TRAIN target (solo naming)
        df_target = load_target(target_path, df_latent.index)
        # alinear labels al índice común de target
        common = df_latent.index.intersection(df_target.index)
        if len(common) == 0:
            raise ValueError("No hay solapamiento de índices entre latent_ae y target.csv")
        # recomputar labels/probs sobre el subset común si hiciera falta
        # (normalmente common == todo train)
        idx_pos = df_latent.index.get_indexer(common)
        labels_common = labels[idx_pos]

        stats = _cluster_stats_from_target(labels_common, df_target.loc[common], cfg.k)
        mapping = build_mapping_from_target(stats)

        # también añadimos un snapshot de stats completo al metrics report
        metrics["cluster_outcome_stats_train"] = stats.to_dict(orient="records")

    else:
        mapping = build_default_mapping(cfg.k)

    out_mapping.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    # Save metrics
    out_metrics.write_text(json.dumps({"config": asdict(cfg), "metrics": metrics}, indent=2, ensure_ascii=False))

    # Console summary
    typer.echo("✅ GMM training finished (TRAIN full).")
    typer.echo(f"   - Window: W{int(window):02d} | embeddings: {latent_path}")
    typer.echo(f"   - N students: {metrics['n_students']} | latent_dim: {metrics['latent_dim']}")
    typer.echo(f"   - k={metrics['k']} cov={metrics['covariance_type']} seed={metrics['seed']}")
    typer.echo(f"   - BIC: {metrics['bic']:.0f} | LogLik(mean): {metrics['log_likelihood_mean']:.4f}")
    typer.echo(f"   - Confidence(mean): {metrics['confidence_mean']:.2%} | doubtful<{cfg.doubtful_threshold}: {metrics['doubtful_pct']:.2%}")
    typer.echo(f"   - min_cluster_pct: {metrics['min_cluster_pct']:.2%}")
    typer.echo(f"🗂  Mapping: {'manual' if mapping_json_in else ('auto(target)' if auto_mapping else 'default')}")
    typer.echo(f"📦 Saved: {out_gmm}")
    typer.echo(f"📦 Saved: {out_scaler}")
    typer.echo(f"🗂  Saved: {out_mapping}")
    typer.echo(f"📄 Saved: {out_metrics}")


if __name__ == "__main__":
    app()
