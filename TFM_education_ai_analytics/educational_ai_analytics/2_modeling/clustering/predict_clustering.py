import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from educational_ai_analytics.config import EMBEDDINGS_DATA_DIR, MODELS_DIR, W_WINDOWS

app = typer.Typer(add_completion=False)

DEFAULT_SPLITS = ("training", "validation", "test")
DEFAULT_WINDOW = int(max(W_WINDOWS)) if W_WINDOWS else 24


@dataclass(frozen=True)
class Paths:
    embeddings_dir: Path = EMBEDDINGS_DATA_DIR
    models_dir: Path = MODELS_DIR
    out_dir: Path = EMBEDDINGS_DATA_DIR

    latent_filename: str = "ae_latent.csv"
    gmm_filename: str = "gmm_ae.joblib"
    scaler_filename: str = "scaler_latent_ae.joblib"
    mapping_filename: str = "cluster_mapping.json"


def _load_latent(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Latent file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.sort_index()


def _load_mapping(path: Path, k: int) -> Dict[str, Dict[str, str]]:
    """
    Expected mapping JSON:
    {
      "0": {"label": "...", "name": "..."},
      ...
    }
    """
    if path.exists():
        mp = json.loads(path.read_text())
        # ensure all keys exist
        for i in range(k):
            mp.setdefault(str(i), {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"})
        return mp
    # default mapping if not present
    return {str(i): {"label": f"CLUSTER_{i}", "name": f"Cluster {i}"} for i in range(k)}


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy per row. p shape (N, K)."""
    return -(p * np.log(p + eps)).sum(axis=1)


def _predict_one_split(
    split: str,
    window: int,
    paths: Paths,
    gmm,
    scaler,
    mapping: Dict[str, Dict[str, str]],
    write_outputs: bool = True,
) -> Optional[Path]:
    latent_path = paths.embeddings_dir / split / f"upto_w{int(window):02d}" / paths.latent_filename
    if not latent_path.exists():
        logger.warning(f"[{split}] No latent file at {latent_path}. Skipping.")
        return None

    df_lat = _load_latent(latent_path)
    X = df_lat.values.astype(np.float32)
    Xs = scaler.transform(X)

    p = gmm.predict_proba(Xs)  # (N,K)
    cluster_id = p.argmax(axis=1).astype(int)
    confidence = p.max(axis=1)

    ent = _entropy(p)
    ent_norm = ent / np.log(p.shape[1])  # [0,1] approx

    # build output DF
    out = pd.DataFrame(index=df_lat.index)
    out.index.name = df_lat.index.name

    out["cluster_id"] = cluster_id
    out["cluster_label"] = [mapping[str(i)]["label"] for i in cluster_id]
    out["cluster_name"] = [mapping[str(i)]["name"] for i in cluster_id]

    # probs
    for j in range(p.shape[1]):
        out[f"p_cluster_{j}"] = p[:, j]

    out["confidence"] = confidence
    out["entropy"] = ent
    out["entropy_norm"] = ent_norm

    # (optional) keep latent dims? usually NO for downstream, but nice for debugging:
    # out = out.join(df_lat.add_prefix("latent_"))

    if not write_outputs:
        logger.info(f"[{split}] Computed segmentation (not saved). shape={out.shape}")
        return None

    out_dir = paths.out_dir / split / f"upto_w{int(window):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segmentation_gmm_ae.csv"
    out.to_csv(out_path)

    logger.success(f"[{split}] ✅ Saved segmentation: {out_path} | N={len(out)} | K={p.shape[1]}")
    return out_path


@app.command()
def main(
    splits: str = typer.Option(
        "training,validation,test",
        help="Comma-separated splits to run (e.g. 'training,validation,test' or 'validation,test').",
    ),
    embeddings_dir: Path = typer.Option(EMBEDDINGS_DATA_DIR, help="Base embeddings dir (contains split subfolders)."),
    window: int = typer.Option(DEFAULT_WINDOW, help="Ventana W a usar (upto_wXX)."),
    latent_filename: str = typer.Option("ae_latent.csv", help="Latent CSV name inside each split folder."),
    models_dir: Path = typer.Option(MODELS_DIR, help="Models directory."),
    gmm_filename: str = typer.Option("gmm_ae.joblib", help="GMM artifact filename."),
    scaler_filename: str = typer.Option("scaler_latent_ae.joblib", help="Scaler artifact filename."),
    mapping_filename: str = typer.Option("cluster_mapping.json", help="Cluster mapping JSON filename."),
    out_dir: Path = typer.Option(
        EMBEDDINGS_DATA_DIR,
        help="Output base directory (default: embeddings, same split/upto_wXX).",
    ),
):
    """
    Production inference of clustering (AE+GMM):
    - Loads ae_latent.csv for each split in upto_wXX
    - Loads scaler_latent_ae.joblib + gmm_ae.joblib
    - Computes p_cluster_0..K-1 + confidence + entropy(+norm) + cluster_name/label
    - Saves segmentation_gmm_ae.csv per split in embeddings/{split}/upto_wXX
    """
    split_list = tuple([s.strip() for s in splits.split(",") if s.strip()])
    if not split_list:
        logger.error("No splits provided.")
        raise typer.Exit(code=1)

    paths = Paths(
        embeddings_dir=embeddings_dir,
        models_dir=models_dir,
        out_dir=out_dir,
        latent_filename=latent_filename,
        gmm_filename=gmm_filename,
        scaler_filename=scaler_filename,
        mapping_filename=mapping_filename,
    )

    gmm_path = paths.models_dir / paths.gmm_filename
    scaler_path = paths.models_dir / paths.scaler_filename
    mapping_path = paths.models_dir / paths.mapping_filename

    if not gmm_path.exists():
        logger.error(f"GMM artifact not found: {gmm_path}")
        raise typer.Exit(code=1)
    if not scaler_path.exists():
        logger.error(f"Scaler artifact not found: {scaler_path}")
        raise typer.Exit(code=1)

    logger.info(f"Loading GMM: {gmm_path}")
    gmm = joblib.load(gmm_path)

    logger.info(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Determine K from gmm
    try:
        k = int(gmm.n_components)
    except Exception:
        # fallback from means_
        k = int(getattr(gmm, "means_").shape[0])

    mapping = _load_mapping(mapping_path, k)

    logger.info(f"Running clustering inference for splits={split_list} | W={window} | K={k}")
    for split in split_list:
        _predict_one_split(
            split,
            window=window,
            paths=paths,
            gmm=gmm,
            scaler=scaler,
            mapping=mapping,
            write_outputs=True,
        )

    logger.success("✅ Done.")


if __name__ == "__main__":
    app()
