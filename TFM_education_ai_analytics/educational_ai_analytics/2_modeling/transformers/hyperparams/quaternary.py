from .schema import TransformerHyperparams


TRANSFORMER_PARAMS = TransformerHyperparams(
	num_classes=4,
	focal_alpha=[0.27, 0.35, 0.15, 0.23],
)