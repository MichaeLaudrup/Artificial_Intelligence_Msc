from dataclasses import dataclass
from typing import Optional


@dataclass
class XAIHyperparams:
	split: str = "validation"
	top_k: int = 10
	shap_bg_size: int = 40
	shap_explain_size: int = 60
	shap_nsamples: int = 60
	seed: int = 42
	with_static: bool = True
	weeks_csv: Optional[str] = None


XAI_PARAMS = XAIHyperparams()
