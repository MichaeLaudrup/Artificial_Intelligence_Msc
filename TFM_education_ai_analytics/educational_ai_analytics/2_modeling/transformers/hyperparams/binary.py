from .schema import TransformerHyperparams


TRANSFORMER_PARAMS = TransformerHyperparams(
	paper_baseline=False,
	binary_mode="success_vs_risk",
	num_classes=2,
	focal_alpha=[0.46, 0.54],
)