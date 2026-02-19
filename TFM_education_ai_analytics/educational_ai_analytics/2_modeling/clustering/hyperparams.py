from dataclasses import dataclass, field
from typing import List

from educational_ai_analytics.config import W_WINDOWS


@dataclass(frozen=True)
class ClusteringParams:
	# Embeddings
	latent_filename: str = "ae_latent.csv"
	split_train: str = "training"
	split_predict: List[str] = field(default_factory=lambda: ["training", "validation", "test"])
	windows: List[int] = field(default_factory=lambda: [int(w) for w in W_WINDOWS])

	# GMM defaults (alineado con notebook)
	n_clusters: int = 5
	gmm_k_range: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
	covariance_type: str = "diag"
	reg_covar: float = 1e-6
	max_iter: int = 300
	n_init: int = 10
	init_params: str = "kmeans"
	random_state: int = 42

	# Selección de K
	use_fixed_k: bool = True
	use_bic: bool = True
	use_aic: bool = False
	marginal_gain_pct_threshold: float = 1.0

	# Calidad/diagnóstico
	confidence_doubtful_threshold: float = 0.60
	silhouette_sample_size: int = 10000

	# Artefactos
	model_prefix: str = "gmm_ae"
	scaler_prefix: str = "scaler_latent_ae"
	mapping_prefix: str = "cluster_mapping"
	metrics_prefix: str = "train_clustering_metrics"
	diagnostics_filename: str = "gmm_k_diagnostics.csv"


CLUSTERING_PARAMS = ClusteringParams()

