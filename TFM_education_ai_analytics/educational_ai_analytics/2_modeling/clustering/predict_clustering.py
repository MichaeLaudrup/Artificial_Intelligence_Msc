import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from educational_ai_analytics.config import EMBEDDINGS_DATA_DIR, CLUSTERING_MODELS_DIR, MODELS_DIR

from .hyperparams import CLUSTERING_PARAMS

app = typer.Typer(add_completion=False)


def _parse_csv_list(value: Optional[str], default_values: List[str]) -> List[str]:
    if not value:
        return list(default_values)
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_windows(value: Optional[str]) -> List[int]:
    if not value:
        return [int(w) for w in CLUSTERING_PARAMS.windows]
    return sorted({int(x.strip()) for x in value.split(",") if x.strip()})


def _latent_path(split: str, window: int, latent_filename: str) -> Path:
    return EMBEDDINGS_DATA_DIR / split / f"upto_w{int(window):02d}" / latent_filename


def _artifact_paths(window: int) -> Dict[str, Path]:
    return {
        "gmm": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}_w{int(window):02d}.joblib",
        "scaler": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}_w{int(window):02d}.joblib",
        "mapping": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}_w{int(window):02d}.json",
    }


def _legacy_artifact_paths() -> Dict[str, Path]:
    return {
        "gmm": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}.joblib",
        "scaler": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}.joblib",
        "mapping": CLUSTERING_MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}.json",
    }


def _very_legacy_artifact_paths() -> Dict[str, Path]:
    return {
        "gmm": MODELS_DIR / f"{CLUSTERING_PARAMS.model_prefix}.joblib",
        "scaler": MODELS_DIR / f"{CLUSTERING_PARAMS.scaler_prefix}.joblib",
        "mapping": MODELS_DIR / f"{CLUSTERING_PARAMS.mapping_prefix}.json",
    }


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_mapping(path: Path, k: int) -> Dict[str, Dict[str, str]]:
    if path.exists():
        mp = json.loads(path.read_text())
        for i in range(k):
            mp.setdefault(str(i), {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"})
        return mp
    return {str(i): {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"} for i in range(k)}


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(probs * np.log(probs + eps)).sum(axis=1)


def _predict_one(split: str, window: int, latent_filename: str, out_filename: str) -> Optional[Path]:
    lat_path = _latent_path(split=split, window=window, latent_filename=latent_filename)
    df_lat = _safe_read_csv(lat_path)
    if df_lat.empty:
        logger.warning(f"⚠️ [{split}] W{window:02d}: embeddings no encontrados/vacíos en {lat_path}")
        return None

    artifacts = _artifact_paths(window)
    if not artifacts["gmm"].exists() or not artifacts["scaler"].exists():
        logger.warning(f"⚠️ [{split}] W{window:02d}: artefactos por ventana no encontrados, intentando legacy...")
        artifacts = _legacy_artifact_paths()

    if not artifacts["gmm"].exists() or not artifacts["scaler"].exists():
        logger.warning(f"⚠️ [{split}] W{window:02d}: legacy en clustering_models no encontrado, intentando models raíz...")
        artifacts = _very_legacy_artifact_paths()

    if not artifacts["gmm"].exists() or not artifacts["scaler"].exists():
        logger.error(f"❌ [{split}] W{window:02d}: no hay modelo/scaler para inferencia")
        return None

    gmm = joblib.load(artifacts["gmm"])
    scaler = joblib.load(artifacts["scaler"])

    k = int(getattr(gmm, "n_components", 0))
    mapping = _load_mapping(artifacts["mapping"], k=k)

    X = df_lat.values.astype(np.float32)
    Xs = scaler.transform(X)

    probs = gmm.predict_proba(Xs)
    labels = probs.argmax(axis=1).astype(int)
    confidence = probs.max(axis=1)
    entropy = _entropy(probs)
    entropy_norm = entropy / np.log(probs.shape[1])

    out = pd.DataFrame(index=df_lat.index)
    out.index.name = df_lat.index.name

    out["cluster_id"] = labels
    out["cluster_label"] = [mapping[str(i)]["label"] for i in labels]
    out["cluster_name"] = [mapping[str(i)]["name"] for i in labels]

    for j in range(probs.shape[1]):
        out[f"p_cluster_{j}"] = probs[:, j]

    out["confidence"] = confidence
    out["entropy"] = entropy
    out["entropy_norm"] = entropy_norm

    out_dir = EMBEDDINGS_DATA_DIR / split / f"upto_w{int(window):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_filename
    out.to_csv(out_path)

    logger.success(f"✅ [{split}] W{window:02d}: {out_path.name} | N={len(out)} | K={probs.shape[1]}")
    return out_path


@app.command()
def main(
    splits: Optional[str] = typer.Option(
        None,
        help="Splits separados por coma. Por defecto usa hyperparams (training,validation,test).",
    ),
    windows: Optional[str] = typer.Option(
        None,
        help="Ventanas separadas por coma (ej: 12,18,24). Por defecto usa W_WINDOWS de config.py.",
    ),
    latent_filename: str = typer.Option(CLUSTERING_PARAMS.latent_filename, help="Nombre del CSV latente de entrada."),
    out_filename: str = typer.Option("segmentation_gmm_ae.csv", help="Nombre del CSV de salida por split/ventana."),
):
    """
    Inferencia GMM multi-ventana:
      - Carga artefactos por ventana (gmm_ae_wXX, scaler_latent_ae_wXX, mapping_wXX)
      - Predice por split y ventana
      - Guarda segmentation_gmm_ae.csv en embeddings/{split}/upto_wXX
    """
    split_list = _parse_csv_list(splits, CLUSTERING_PARAMS.split_predict)
    window_list = _parse_windows(windows)

    logger.info(f"🔮 Clustering PREDICT | splits={split_list} | windows={window_list}")

    generated = 0
    for w in window_list:
        for split in split_list:
            out_path = _predict_one(
                split=split,
                window=int(w),
                latent_filename=latent_filename,
                out_filename=out_filename,
            )
            if out_path is not None:
                generated += 1

    if generated == 0:
        raise RuntimeError("No se generó ninguna segmentación. Revisa embeddings/artefactos/modelos.")

    logger.success(f"🎯 Segmentaciones generadas: {generated}")


if __name__ == "__main__":
    app()
