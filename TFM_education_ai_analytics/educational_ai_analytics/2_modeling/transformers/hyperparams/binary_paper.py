from .schema import TransformerHyperparams


TRANSFORMER_PARAMS = TransformerHyperparams(
	paper_baseline=True,
	binary_mode="paper",
	num_classes=2,
	focal_alpha=[0.27, 0.73],
)