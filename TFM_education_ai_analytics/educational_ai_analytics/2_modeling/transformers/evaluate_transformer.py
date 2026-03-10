import json
import inspect
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
	from .hyperparams import TRANSFORMER_PARAMS
	from .report_paths import migrate_legacy_transformer_reports, normalize_binary_mode as normalize_binary_mode_shared, resolve_report_dir, resolve_report_scope_name
	from .train_transformer import (
		_available_transformer_weeks,
		_normalize_binary_mode,
		_suppress_low_level_cuda_noise,
		load_and_prepare_split,
	)
	from .transformer_GLU_classifier import GLULayer, GLUTransformerClassifier, TransformerEncoderBlock
	from .utils.thresholding import select_binary_threshold_with_constraints
	from .utils.training_config import build_runtime_config_from_cli
except ImportError:
	from hyperparams import TRANSFORMER_PARAMS
	from report_paths import migrate_legacy_transformer_reports, normalize_binary_mode as normalize_binary_mode_shared, resolve_report_dir, resolve_report_scope_name
	from train_transformer import (
		_available_transformer_weeks,
		_normalize_binary_mode,
		_suppress_low_level_cuda_noise,
		load_and_prepare_split,
	)
	from transformer_GLU_classifier import GLULayer, GLUTransformerClassifier, TransformerEncoderBlock
	from utils.thresholding import select_binary_threshold_with_constraints
	from utils.training_config import build_runtime_config_from_cli

from educational_ai_analytics.tf_runtime import configure_tensorflow_runtime, resolve_execution_device

app = typer.Typer(help="Evaluación standalone del transformer sobre test.")


def _resolve_target_tag(num_classes: int, paper_baseline: bool, binary_mode: Optional[str]) -> tuple[str, Optional[str]]:
	if int(num_classes) == 2:
		resolved_mode = normalize_binary_mode_shared(paper_baseline=paper_baseline, binary_mode=binary_mode)
		return f"{num_classes}clases_{resolved_mode}", resolved_mode
	return f"{num_classes}clases", None


def _hyperparams_module_path() -> str:
	module = sys.modules.get(TRANSFORMER_PARAMS.__class__.__module__)
	module_file = getattr(module, "__file__", None)
	return str(module_file) if module_file else inspect.getfile(TRANSFORMER_PARAMS.__class__)


def _resolve_eval_week(base_npz: Path, requested_week: Optional[int], default_week: int) -> int:
	available_weeks = _available_transformer_weeks(base_npz, ["validation", "test"])
	if not available_weeks:
		raise FileNotFoundError(
			f"No hay features transformer disponibles en {base_npz} para evaluación en los splits ['validation', 'test']"
		)

	if requested_week is None:
		return default_week if default_week in available_weeks else available_weeks[-1]

	if requested_week in available_weeks:
		return requested_week

	fallback_week = next((week for week in available_weeks if week >= requested_week), available_weeks[-1])
	logger.warning(
		f"⚠️ upto_week={requested_week} no está disponible para evaluación. "
		f"Ventanas disponibles: {available_weeks}. Usando upto_week={fallback_week}."
	)
	return fallback_week


def _is_accumulated_name(col: str) -> bool:
	col = str(col)
	accumulated_prefixes = (
		"clicks_",
		"weeks_since_",
		"streak_",
		"total_weighted_",
		"last_week_",
		"momentum",
		"regularity",
		"weekend_share",
		"distinct_activity",
		"recency_",
		"engagement_",
		"pass_ratio",
		"late_ratio",
		"submission_",
		"avg_score",
		"active_weeks",
	)
	return col.startswith(accumulated_prefixes)


def _is_cluster_name(col: str) -> bool:
	col = str(col)
	return col.startswith("p_cluster_") or col == "entropy_norm"


def _cluster_drop_mask(feature_names: np.ndarray, feature_sources: np.ndarray, n_features: int, cluster_dim: int) -> np.ndarray:
	if n_features == 0:
		return np.zeros((0,), dtype=bool)

	if feature_sources.size == n_features:
		src = np.array(feature_sources).astype(str)
		return src == "cluster"

	if feature_names.size == n_features:
		names = np.array(feature_names).astype(str)
		return np.array([_is_cluster_name(column) for column in names], dtype=bool)

	fallback = np.zeros((n_features,), dtype=bool)
	if cluster_dim > 0:
		fallback[: min(cluster_dim, n_features)] = True
	return fallback


def _filter_static_block(
	x_static: Optional[np.ndarray],
	cluster_dim: int,
	feature_names: np.ndarray,
	feature_sources: np.ndarray,
	keep_clusters: bool,
	keep_accumulated: bool,
	split_name: str,
) -> Optional[np.ndarray]:
	if x_static is None:
		return None

	n_features = x_static.shape[1]
	keep_mask = np.ones(n_features, dtype=bool)
	drop_cluster_mask = _cluster_drop_mask(feature_names, feature_sources, n_features, cluster_dim)

	if not keep_clusters:
		keep_mask &= ~drop_cluster_mask

	if not keep_accumulated:
		if feature_sources.size == n_features:
			src = np.array(feature_sources).astype(str)
			keep_mask &= src != "accumulated_uptow"
		elif feature_names.size == n_features:
			names = np.array(feature_names).astype(str)
			drop_acc = np.array([_is_accumulated_name(column) for column in names], dtype=bool)
			keep_mask &= ~drop_acc
		else:
			logger.warning(
				f"⚠️ [{split_name}] No hay metadata de columnas estáticas; no se puede filtrar accumulated_uptow de forma segura"
			)

	logger.info(
		f"[{split_name}] Static ablation -> total={n_features} | kept={int(keep_mask.sum())} | removed={int((~keep_mask).sum())}"
	)

	if keep_mask.sum() == 0:
		return np.zeros((x_static.shape[0], 0), dtype=x_static.dtype)
	return x_static[:, keep_mask]


def _load_history_entries(history_path: Path) -> list[dict]:
	if not history_path.exists():
		return []
	try:
		payload = json.loads(history_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		logger.warning(f"Historial corrupto en {history_path}: {exc}")
		return []
	return payload if isinstance(payload, list) else []


def _select_latest_matching_entry(
	entries: list[dict],
	upto_week: int,
	num_classes: int,
	expected_binary_mode: Optional[str],
	paper_baseline: bool,
) -> Optional[dict]:
	for entry in reversed(entries):
		hyperparams = entry.get("hyperparameters", {})
		try:
			entry_week = int(hyperparams.get("upto_week", upto_week))
			entry_num_classes = int(hyperparams.get("num_classes", num_classes))
		except (TypeError, ValueError):
			continue

		if entry_week != int(upto_week) or entry_num_classes != int(num_classes):
			continue

		if int(num_classes) == 2:
			try:
				entry_mode = _normalize_binary_mode(
					paper_baseline=bool(hyperparams.get("paper_baseline", paper_baseline)),
					binary_mode=hyperparams.get("binary_mode", expected_binary_mode),
				)
			except ValueError:
				continue
			if entry_mode != expected_binary_mode:
				continue

		return entry
	return None


def _custom_objects() -> dict:
	return {
		"GLUTransformerClassifier": GLUTransformerClassifier,
		"TransformerEncoderBlock": TransformerEncoderBlock,
		"GLULayer": GLULayer,
	}


def _save_test_metrics_to_history(
	history_path: Path,
	runtime_cfg,
	selected_binary_mode: Optional[str],
	test_metrics: dict,
	selected_threshold: Optional[float],
) -> None:
	entries = _load_history_entries(history_path)
	latest_entry = _select_latest_matching_entry(
		entries=entries,
		upto_week=runtime_cfg.upto_week,
		num_classes=runtime_cfg.num_classes,
		expected_binary_mode=selected_binary_mode,
		paper_baseline=runtime_cfg.paper_baseline,
	)
	metadata = {
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
	}

	if latest_entry is None:
		latest_entry = {
			"timestamp": metadata["timestamp"],
			"hyperparameters": asdict(runtime_cfg),
		}
		entries.append(latest_entry)

	latest_entry["test_metrics"] = test_metrics
	latest_entry["test_evaluation"] = metadata
	history_path.parent.mkdir(parents=True, exist_ok=True)
	history_path.write_text(json.dumps(entries, indent=4), encoding="utf-8")


@app.command()
def evaluate(
	config_json: Optional[Path] = typer.Option(None, help="Ruta a JSON con TrainingConfig para inyección externa"),
	metrics_out: Optional[Path] = typer.Option(None, help="Ruta de salida JSON con métricas de test"),
):
	"""Evalúa el último modelo entrenado sobre test sin contaminar el entrenamiento."""
	runtime_cfg = build_runtime_config_from_cli(config_json=config_json)
	if config_json is not None:
		logger.info(f"📥 Config inyectada desde JSON: {config_json}")

	base_npz = Path("/workspace/TFM_education_ai_analytics/data/6_transformer_features")
	runtime_cfg.upto_week = _resolve_eval_week(base_npz, runtime_cfg.upto_week, TRANSFORMER_PARAMS.upto_week)
	target_tag, selected_binary_mode = _resolve_target_tag(
		num_classes=runtime_cfg.num_classes,
		paper_baseline=runtime_cfg.paper_baseline,
		binary_mode=runtime_cfg.binary_mode,
	)
	logger.info("🧭 Hyperparams cargados desde: {}", _hyperparams_module_path())
	logger.info(
		"🧭 Config efectiva evaluación -> upto_week={} | num_classes={} | target_tag={} | binary_mode={} | paper_baseline={}",
		runtime_cfg.upto_week,
		runtime_cfg.num_classes,
		target_tag,
		selected_binary_mode if selected_binary_mode is not None else "ignored",
		runtime_cfg.paper_baseline,
	)

	migrate_legacy_transformer_reports(Path("/workspace/TFM_education_ai_analytics/reports/transformer_training"))
	reports_dir = resolve_report_dir(
		Path("/workspace/TFM_education_ai_analytics/reports/transformer_training"),
		num_classes=runtime_cfg.num_classes,
		paper_baseline=runtime_cfg.paper_baseline,
		binary_mode=runtime_cfg.binary_mode,
	) / f"week_{runtime_cfg.upto_week}"
	history_file = runtime_cfg.history_filename or f"experiments_history_{target_tag}.json"
	history_path = reports_dir / history_file
	model_path = Path(f"/workspace/TFM_education_ai_analytics/models/transformers/transformer_uptoW{runtime_cfg.upto_week}_{target_tag}.keras")
	if not model_path.exists():
		raise FileNotFoundError(f"No se encuentra el modelo para evaluar: {model_path}")
	if not history_path.exists():
		raise FileNotFoundError(f"No se encuentra el history asociado al modelo: {history_path}")

	X_val_seq, val_mask_pad, val_mask_activity, X_val_stat, y_val, val_ids, val_cluster_dim, val_static_names, val_static_sources = load_and_prepare_split(
		base_npz,
		"validation",
		runtime_cfg.upto_week,
		runtime_cfg.num_classes,
		runtime_cfg.paper_baseline,
		runtime_cfg.with_static,
		runtime_cfg.binary_mode,
	)
	X_test_seq, test_mask_pad, test_mask_activity, X_test_stat, y_test, test_ids, test_cluster_dim, test_static_names, test_static_sources = load_and_prepare_split(
		base_npz,
		"test",
		runtime_cfg.upto_week,
		runtime_cfg.num_classes,
		runtime_cfg.paper_baseline,
		runtime_cfg.with_static,
		runtime_cfg.binary_mode,
	)

	use_static_in_model = runtime_cfg.with_static
	if use_static_in_model and (not runtime_cfg.use_clustering_features or not runtime_cfg.accumulated_uptow):
		X_val_stat = _filter_static_block(
			X_val_stat,
			val_cluster_dim,
			val_static_names,
			val_static_sources,
			keep_clusters=runtime_cfg.use_clustering_features,
			keep_accumulated=runtime_cfg.accumulated_uptow,
			split_name="validation",
		)
		X_test_stat = _filter_static_block(
			X_test_stat,
			test_cluster_dim,
			test_static_names,
			test_static_sources,
			keep_clusters=runtime_cfg.use_clustering_features,
			keep_accumulated=runtime_cfg.accumulated_uptow,
			split_name="test",
		)
		if X_val_stat is None or X_val_stat.shape[1] == 0:
			use_static_in_model = False
			X_val_stat = None
			X_test_stat = None

	final_validation_set = [
		X_val_seq.astype(np.float32),
		val_mask_pad.astype(np.int32),
		val_mask_activity.astype(np.int32),
	]
	final_test_set = [
		X_test_seq.astype(np.float32),
		test_mask_pad.astype(np.int32),
		test_mask_activity.astype(np.int32),
	]
	if use_static_in_model:
		final_validation_set.append(X_val_stat.astype(np.float32))
		final_test_set.append(X_test_stat.astype(np.float32))

	execution_device = resolve_execution_device(TRANSFORMER_PARAMS.execution_device)
	logger.info(f"[{execution_device.upper()}] Initializing runtime environment for standalone test evaluation...")
	configure_tensorflow_runtime(tf, execution_device, logger)

	model = tf.keras.models.load_model(model_path, custom_objects=_custom_objects(), compile=False)

	with _suppress_low_level_cuda_noise():
		y_probs_val = model.predict(final_validation_set, verbose=0)
		y_probs_test = model.predict(final_test_set, verbose=0)

	y_probs_val = np.asarray(y_probs_val, dtype=np.float32)
	y_probs_test = np.asarray(y_probs_test, dtype=np.float32)
	selected_threshold = None

	if runtime_cfg.num_classes == 2 and runtime_cfg.tune_threshold:
		best_thr, candidates = select_binary_threshold_with_constraints(
			y_true=y_val,
			p_pos=y_probs_val[:, 1],
			acc_min=runtime_cfg.threshold_acc_min,
			prec_min=runtime_cfg.threshold_prec_min,
			objective=runtime_cfg.threshold_objective,
			t_min=runtime_cfg.threshold_min,
			t_max=runtime_cfg.threshold_max,
			n_points=runtime_cfg.threshold_points,
		)
		if best_thr is None:
			selected_threshold = float(runtime_cfg.threshold_fallback)
			logger.warning(
				f"⚠️ Threshold tuning sin candidatos factibles. Usando fallback={selected_threshold:.3f}"
			)
		else:
			selected_threshold = float(best_thr["threshold"])
			logger.info(
				f"🎚️ Threshold seleccionado desde validation | candidates={len(candidates)} | threshold={selected_threshold:.3f}"
			)

	if runtime_cfg.num_classes == 2:
		thr_to_use = selected_threshold if selected_threshold is not None else 0.5
		y_pred_val = (y_probs_val[:, 1] >= thr_to_use).astype(int)
		y_pred_test = (y_probs_test[:, 1] >= thr_to_use).astype(int)
		test_auc = float(roc_auc_score(y_test, y_probs_test[:, 1]))
		test_precision = float(precision_score(y_test, y_pred_test, pos_label=1, average="binary"))
		test_recall = float(recall_score(y_test, y_pred_test, pos_label=1, average="binary"))
		test_f1 = float(f1_score(y_test, y_pred_test, pos_label=1, average="binary"))
		test_top2_acc = None
	else:
		y_pred_val = np.argmax(y_probs_val, axis=1)
		y_pred_test = np.argmax(y_probs_test, axis=1)
		test_auc = float(roc_auc_score(y_test, y_probs_test, multi_class="ovr"))
		test_precision = float(precision_score(y_test, y_pred_test, average="macro"))
		test_recall = float(recall_score(y_test, y_pred_test, average="macro"))
		test_f1 = float(f1_score(y_test, y_pred_test, average="macro"))
		test_top2_acc = None
		if runtime_cfg.num_classes == 4:
			top2_idx_test = np.argsort(y_probs_test, axis=1)[:, -2:]
			test_top2_acc = float(np.mean(np.any(top2_idx_test == y_test.reshape(-1, 1), axis=1)))

	# === Guardar Predicciones para Stacking ===
	preds_root_dir = Path("/workspace/TFM_education_ai_analytics/data/7_model_predictions")
	preds_scope_name = resolve_report_scope_name(
		num_classes=runtime_cfg.num_classes,
		paper_baseline=runtime_cfg.paper_baseline,
		binary_mode=runtime_cfg.binary_mode,
	)
	preds_out_dir = preds_root_dir / preds_scope_name
	
	# Mapeo de nombres de clases para que el CSV sea más legible
	class_names = []
	if runtime_cfg.num_classes == 2:
		if runtime_cfg.paper_baseline:
			class_names = ["Pass/Dist", "Withdrawn"]
		elif runtime_cfg.binary_mode == "original":
			class_names = ["Pass/Dist", "Fail"]
		else:
			class_names = ["Pass/Dist", "Fail/Withdrawn"]
	elif runtime_cfg.num_classes == 3:
		class_names = ["Fail", "Withdrawn", "Pass/Dist"]
	elif runtime_cfg.num_classes == 4:
		class_names = ["Fail", "Withdrawn", "Pass", "Distinction"]
	else:
		class_names = [f"Class_{i}" for i in range(runtime_cfg.num_classes)]

	# validation
	val_out_dir = preds_out_dir / "validation"
	val_out_dir.mkdir(parents=True, exist_ok=True)
	
	val_df = pd.DataFrame({
		"id_student": val_ids.astype(str),
		"classification_scope": preds_scope_name,
		"target_tag": target_tag,
	})
	for c, name in enumerate(class_names):
		clean_name = name.replace("/", "_").replace(" ", "_").lower()
		val_df[f"prob_{clean_name}"] = y_probs_val[:, c]
	val_df["pred_class_id"] = y_pred_val
	val_df["pred_class_name"] = [class_names[p] for p in y_pred_val]
	val_df["true_class_id"] = y_val
	val_df["true_class_name"] = [class_names[t] for t in y_val]
	val_df.to_csv(val_out_dir / f"transformer_preds_uptW{runtime_cfg.upto_week}.csv", index=False)
	
	# test
	test_out_dir = preds_out_dir / "test"
	test_out_dir.mkdir(parents=True, exist_ok=True)
	
	test_df = pd.DataFrame({
		"id_student": test_ids.astype(str),
		"classification_scope": preds_scope_name,
		"target_tag": target_tag,
	})
	for c, name in enumerate(class_names):
		clean_name = name.replace("/", "_").replace(" ", "_").lower()
		test_df[f"prob_{clean_name}"] = y_probs_test[:, c]
	test_df["pred_class_id"] = y_pred_test
	test_df["pred_class_name"] = [class_names[p] for p in y_pred_test]
	test_df["true_class_id"] = y_test
	test_df["true_class_name"] = [class_names[t] for t in y_test]
	test_df.to_csv(test_out_dir / f"transformer_preds_uptW{runtime_cfg.upto_week}.csv", index=False)
	
	logger.info(f"💾 Predicciones de Transformer guardadas en: {preds_out_dir}")

	test_metrics = {
		"accuracy": float(np.mean(y_pred_test == y_test)),
		"balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_test)),
		"precision": test_precision,
		"recall": test_recall,
		"f1_score": test_f1,
		"auc_ovr": test_auc,
	}
	if test_top2_acc is not None:
		test_metrics["top_2_acc"] = float(test_top2_acc)

	_save_test_metrics_to_history(
		history_path=history_path,
		runtime_cfg=runtime_cfg,
		selected_binary_mode=selected_binary_mode,
		test_metrics=test_metrics,
		selected_threshold=selected_threshold,
	)
	logger.info(f"✅ Test metrics guardadas en history: {history_path}")

	if metrics_out is not None:
		payload = {
			"config": asdict(runtime_cfg),
			"selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
			"test_metrics": test_metrics,
		}
		metrics_out.parent.mkdir(parents=True, exist_ok=True)
		metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
		logger.info(f"💾 Metrics JSON guardado en: {metrics_out}")


if __name__ == "__main__":
	app()