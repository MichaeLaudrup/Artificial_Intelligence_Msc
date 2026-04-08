from dataclasses import dataclass
from typing import Optional


@dataclass
class XAIHyperparams:
	split: str = "test"
	top_k: int = 6
	num_classes: int = 2
	paper_baseline: bool = True
	binary_mode: str = "paper"
	device: str = "gpu"  
	
	
	
	
	
	
	
	shap_bg_size: int = 100
	shap_explain_size: int = 24
	shap_nsamples: int = 1536
	seed: int = 42
	with_static: bool = True
	weeks_csv: Optional[str] = None


XAI_PARAMS = XAIHyperparams()
