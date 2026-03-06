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
    upto_week: Optional[int] = None
    num_classes: int = TRANSFORMER_PARAMS.num_classes
    paper_baseline: bool = TRANSFORMER_PARAMS.paper_baseline
    binary_mode: Optional[str] = TRANSFORMER_PARAMS.binary_mode
    batch_size: int = TRANSFORMER_PARAMS.batch_size
    with_static: bool = TRANSFORMER_PARAMS.with_static
    use_clustering_features: bool = TRANSFORMER_PARAMS.use_clustering_features
    accumulated_uptow: bool = TRANSFORMER_PARAMS.accumulated_uptow
    eval_test: bool = TRANSFORMER_PARAMS.eval_test
    history_filename: Optional[str] = TRANSFORMER_PARAMS.history_filename
    latent_d: Optional[int] = None
    num_heads: Optional[int] = None
    ff_dim: Optional[int] = None
    dropout: Optional[float] = None
    num_layers: Optional[int] = None
    learning_rate: Optional[float] = None
    focal_gamma: Optional[float] = None
    focal_alpha_pos: Optional[float] = None
    static_hidden_dim: Optional[int] = None
    head_hidden_dim: Optional[int] = None
    reduce_lr_patience: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    seed: Optional[int] = TRANSFORMER_PARAMS.seed
    tune_threshold: bool = TRANSFORMER_PARAMS.tune_threshold
    threshold_acc_min: float = TRANSFORMER_PARAMS.threshold_acc_min
    threshold_prec_min: float = TRANSFORMER_PARAMS.threshold_prec_min
    threshold_objective: str = TRANSFORMER_PARAMS.threshold_objective
    threshold_min: float = TRANSFORMER_PARAMS.threshold_min
    threshold_max: float = TRANSFORMER_PARAMS.threshold_max
    threshold_points: int = TRANSFORMER_PARAMS.threshold_points
    threshold_fallback: float = TRANSFORMER_PARAMS.threshold_fallback
    fast_search: bool = TRANSFORMER_PARAMS.fast_search
    run_compare: bool = TRANSFORMER_PARAMS.run_compare


def load_config_from_json(config_path: Path, base_cfg: TrainingConfig) -> TrainingConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe config_json: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    merged = asdict(base_cfg)
    merged.update(payload)
    return TrainingConfig(**merged)


def build_runtime_config_from_cli(cli_values: Optional[dict] = None, config_json: Optional[Path] = None) -> TrainingConfig:
    cfg_fields = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in (cli_values or {}).items() if k in cfg_fields}
    runtime_cfg = TrainingConfig(**filtered)
    if config_json is not None:
        runtime_cfg = load_config_from_json(config_json, runtime_cfg)
    return runtime_cfg
