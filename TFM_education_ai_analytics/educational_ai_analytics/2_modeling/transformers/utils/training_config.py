from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json

try:
    from ..hyperparams import TRANSFORMER_PARAMS
except ImportError:
    from hyperparams import TRANSFORMER_PARAMS


@dataclass
class TrainingConfig:
    upto_week: int = TRANSFORMER_PARAMS.upto_week
    num_classes: int = 2
    paper_baseline: bool = True
    binary_mode: Optional[str] = TRANSFORMER_PARAMS.binary_mode
    batch_size: int = 64
    with_static: bool = True
    use_clustering_features: bool = TRANSFORMER_PARAMS.use_clustering_features
    accumulated_uptow: bool = TRANSFORMER_PARAMS.accumulated_uptow
    eval_test: bool = False
    history_filename: Optional[str] = None
    latent_d: Optional[int] = None
    num_heads: Optional[int] = None
    ff_dim: Optional[int] = None
    dropout: Optional[float] = None
    num_layers: Optional[int] = None
    learning_rate: Optional[float] = None
    focal_gamma: Optional[float] = None
    focal_alpha_pos: Optional[float] = 0.7
    static_hidden_dim: Optional[int] = None
    head_hidden_dim: Optional[int] = None
    reduce_lr_patience: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    seed: Optional[int] = None
    tune_threshold: bool = False
    threshold_acc_min: float = 0.80
    threshold_prec_min: float = 0.67
    threshold_objective: str = "f1"
    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_points: int = 301
    threshold_fallback: float = 0.50
    fast_search: bool = False
    run_compare: bool = True


def load_config_from_json(config_path: Path, base_cfg: TrainingConfig) -> TrainingConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config_json: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    merged = asdict(base_cfg)
    merged.update(payload)
    return TrainingConfig(**merged)


def build_runtime_config_from_cli(cli_values: dict, config_json: Optional[Path] = None) -> TrainingConfig:
    cfg_fields = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cli_values.items() if k in cfg_fields}
    runtime_cfg = TrainingConfig(**filtered)
    if config_json is not None:
        runtime_cfg = load_config_from_json(config_json, runtime_cfg)
    return runtime_cfg
