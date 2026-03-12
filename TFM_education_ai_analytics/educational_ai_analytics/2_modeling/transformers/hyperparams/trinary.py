from .schema import TransformerHyperparams


TRANSFORMER_PARAMS = TransformerHyperparams(
	num_classes=3,
	focal_gamma=2.0,
	focal_alpha=[0.414, 0.413, 0.174],
	ff_dim=256,
	num_layers=2,
)