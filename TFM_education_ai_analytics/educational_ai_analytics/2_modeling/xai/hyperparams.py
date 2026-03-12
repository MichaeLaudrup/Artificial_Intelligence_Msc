from dataclasses import dataclass
from typing import Optional


@dataclass
class XAIHyperparams:
	split: str = "test"
	top_k: int = 6
	num_classes: int = 2
	paper_baseline: bool = True
	binary_mode: str = "paper"
	device: str = "gpu"  # gpu | cpu
	# Presets SHAP recomendados:
	# - Opcion moderada: shap_bg_size=80, shap_explain_size=20, shap_nsamples=768
	# - Opcion equilibrada: shap_bg_size=100, shap_explain_size=24, shap_nsamples=1024
	# - Opcion robusta para resultados de informe: shap_bg_size=128, shap_explain_size=32, shap_nsamples=2048
	# - Opcion rapida para comprobar estabilidad: shap_bg_size=64, shap_explain_size=16, shap_nsamples=512
	# Por defecto dejamos una opcion moderada porque las ventanas tardias crecen mucho en coste.
	shap_bg_size: int = 80
	shap_explain_size: int = 20
	shap_nsamples: int = 768
	seed: int = 42
	with_static: bool = True
	weeks_csv: Optional[str] = None


XAI_PARAMS = XAIHyperparams()
