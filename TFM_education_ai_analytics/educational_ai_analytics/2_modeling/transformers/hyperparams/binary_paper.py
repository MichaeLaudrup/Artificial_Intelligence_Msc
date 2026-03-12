from .schema import TransformerHyperparams


TRANSFORMER_PARAMS = TransformerHyperparams(
	paper_baseline=True,
	binary_mode="paper",
	num_classes=2,
	focal_alpha=[0.30, 0.70],
	tune_threshold=True,
	threshold_objective="balanced_accuracy",
	threshold_acc_min=0.70,
	threshold_prec_min=0.55,
)