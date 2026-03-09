import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

XAI_PARAMS = getattr(importlib.import_module("educational_ai_analytics.2_modeling.xai.hyperparams"), "XAI_PARAMS")

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def _resolve_xai_device() -> str:
	configured_device = str(getattr(XAI_PARAMS, "device", "cpu")).strip().lower()
	if configured_device not in {"cpu", "gpu"}:
		configured_device = "cpu"
	env_override = os.environ.get("XAI_DISABLE_GPU")
	if env_override is not None:
		return "cpu" if env_override == "1" else "gpu"
	return configured_device


if _resolve_xai_device() == "cpu":
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import typer

from educational_ai_analytics.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, W_WINDOWS
OULAD_ACTIVITY_LABELS = {
	"homepage": "Inicio",
	"oucontent": "Tema",
	"forumng": "Foro",
	"quiz": "Prueba",
	"subpage": "Sub-pag",
	"resource": "Recurso",
	"url": "Enlace",
	"ouelluminate": "Clase",
	"externalquiz": "TestExt",
	"questionnaire": "Sondeo",
	"oucollaborate": "Equipo",
	"ouwiki": "Wiki",
	"page": "Hoja",
	"dualpane": "Panel",
	"htmlactivity": "Web",
	"sharedsubpage": "Hoj-Com",
	"repeatactivity": "Repaso",
	"dataplus": "Datos",
	"glossary": "Lexico",
	"folder": "Carpeta",
}

STATIC_FEATURE_LABELS = {
	"highest_education": "Estud",
	"studied_credits": "Creds",
	"prestart_intensity": "Preint",
	"imd_band": "Socio",
	"clicks_subpage": "Subpag",
	"top1_share": "Top1",
	"prestart_active_days": "Predias",
	"prestart_clicks_total": "Preclk",
	"api_index": "IndAPI",
	"avg_score": "NotaM",
	"weeks_since_last_submission": "SemEnt",
	"score_slope": "PendNot",
	"last_week_clicks_weighted": "ClkUlt",
	"clicks_dualpane": "Panel",
	"streak_final": "RachaF",
	"late_ratio": "Tardio",
	"early_weeks_ratio": "Tempr",
	"weeks_since_last_activity": "SemAct",
	"active_ratio_uptoW": "ActRat",
	"submission_count": "NumEnt",
	"curiosity_index": "Curios",
	"pass_ratio": "Aprob",
	"active_weeks": "SemUso",
}

_report_paths = importlib.import_module("educational_ai_analytics.2_modeling.transformers.report_paths")
migrate_legacy_transformer_reports = getattr(_report_paths, "migrate_legacy_transformer_reports")
normalize_binary_mode_shared = getattr(_report_paths, "normalize_binary_mode")
resolve_report_dir = getattr(_report_paths, "resolve_report_dir")
infer_report_scope_from_name = getattr(_report_paths, "infer_report_scope_from_name")

app = typer.Typer(help="Explicabilidad SHAP productiva para Transformers.")
set_style = getattr(importlib.import_module("educational_ai_analytics.3_plots.style"), "set_style")
set_style()

TRANSFORMER_MODELS_ROOT = MODELS_DIR / "transformers"
TRANSFORMER_REPORTS_ROOT = REPORTS_DIR / "transformer_training"
TRANSFORMER_FEATURES_ROOT = DATA_DIR / "6_transformer_features"
XAI_REPORTS_ROOT = REPORTS_DIR / "XAI"


@app.callback()
def main():
	"""Comandos XAI para reportes SHAP de transformers."""


def _load_active_transformer_params():
	try:
		module = importlib.import_module("educational_ai_analytics.2_modeling.transformers.hyperparams")
		return getattr(module, "TRANSFORMER_PARAMS", None)
	except Exception as exc:
		logger.warning(f"No se pudieron cargar los hyperparams activos del transformer: {exc}")
		return None


def _map_oulad_activity_label(name: str) -> str:
	raw_name = str(name).strip()
	return OULAD_ACTIVITY_LABELS.get(raw_name, raw_name)


def _map_static_feature_label(name: str) -> str:
	raw_name = str(name).strip()
	return STATIC_FEATURE_LABELS.get(raw_name, raw_name)


def _migrate_legacy_xai_reports(report_root: Path) -> list[str]:
	report_root = Path(report_root)
	if not report_root.exists():
		return []

	moved_paths: list[str] = []

	for path in [child for child in report_root.iterdir() if child.is_file()]:
		scope_name = infer_report_scope_from_name(path.name)
		if scope_name is None:
			continue
		dst_dir = report_root / scope_name
		dst_dir.mkdir(parents=True, exist_ok=True)
		dst_path = dst_dir / path.name
		if dst_path.exists():
			path.unlink()
		else:
			shutil.move(str(path), str(dst_path))
		moved_paths.append(str(dst_path.relative_to(report_root)))

	for week_dir in [child for child in report_root.glob("week_*") if child.is_dir()]:
		for path in [child for child in week_dir.iterdir() if child.is_file()]:
			scope_name = infer_report_scope_from_name(path.name)
			if scope_name is None:
				continue
			dst_dir = report_root / scope_name / week_dir.name
			dst_dir.mkdir(parents=True, exist_ok=True)
			dst_path = dst_dir / path.name
			if dst_path.exists():
				path.unlink()
			else:
				shutil.move(str(path), str(dst_path))
			moved_paths.append(str(dst_path.relative_to(report_root)))
		if not any(week_dir.iterdir()):
			week_dir.rmdir()

	return moved_paths


def _normalize_binary_mode(paper_baseline: bool, binary_mode: Optional[str]) -> str:
	return normalize_binary_mode_shared(paper_baseline=paper_baseline, binary_mode=binary_mode)


def _resolve_target_tag(num_classes: int, paper_baseline: bool, binary_mode: Optional[str]) -> tuple[str, Optional[str]]:
	if int(num_classes) == 2:
		resolved_mode = _normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
		return f"{num_classes}clases_{resolved_mode}", resolved_mode
	return f"{num_classes}clases", None


def _parse_weeks_csv(weeks_csv: Optional[str], fallback_weeks: Iterable[int]) -> list[int]:
	if weeks_csv is None or not str(weeks_csv).strip():
		return sorted({int(week) for week in fallback_weeks})

	parsed = []
	for token in str(weeks_csv).split(","):
		token = token.strip()
		if not token:
			continue
		if not token.isdigit():
			raise ValueError(f"Semana inválida en weeks_csv: {token}")
		parsed.append(int(token))
	return sorted(set(parsed))


def _discover_available_feature_weeks(data_root: Path, split: str) -> set[int]:
	split_dir = data_root / split
	if not split_dir.exists():
		return set()
	return {
		int(path.stem.removeprefix("transformer_uptoW"))
		for path in split_dir.glob("transformer_uptoW*.npz")
		if path.stem.removeprefix("transformer_uptoW").isdigit()
	}


def _resolve_weeks(weeks_csv: Optional[str], split: str, data_root: Path) -> list[int]:
	available = _discover_available_feature_weeks(data_root=data_root, split=split)
	preferred = _parse_weeks_csv(weeks_csv, W_WINDOWS)
	if not available:
		return preferred
	return [week for week in preferred if week in available]


def _load_transformer_custom_objects():
	from ..transformers.transformer_GLU_classifier import (  # pylint: disable=import-outside-toplevel
		GLULayer,
		GLUTransformerClassifier,
		TransformerEncoderBlock,
	)

	return {
		"GLUTransformerClassifier": GLUTransformerClassifier,
		"TransformerEncoderBlock": TransformerEncoderBlock,
		"GLULayer": GLULayer,
	}


def _load_history_entries(history_path: Path) -> list[dict]:
	if not history_path.exists():
		return []
	try:
		payload = json.loads(history_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		logger.warning(f"Historial corrupto en {history_path}: {exc}")
		return []
	return payload if isinstance(payload, list) else []


def _cleanup_residual_xai_files(report_root: Path, legacy_root: Path) -> None:
	for pattern in ("xai_*", "attention_*"):
		for path in report_root.glob(pattern):
			if path.is_file():
				path.unlink(missing_ok=True)
		for week_dir in report_root.glob("week_*"):
			if week_dir.is_dir():
				for path in week_dir.glob(pattern):
					if path.is_file():
						path.unlink(missing_ok=True)

	for pattern in ("xai_*", "attention_*"):
		for path in legacy_root.glob(pattern):
			if path.is_file():
				path.unlink(missing_ok=True)
		for week_dir in legacy_root.glob("week_*"):
			if week_dir.is_dir():
				for path in week_dir.glob(pattern):
					if path.is_file():
						path.unlink(missing_ok=True)


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


def _week_artifacts(
	upto_week: int,
	num_classes: int,
	paper_baseline: bool,
	binary_mode: Optional[str],
	models_root: Path,
	reports_root: Path,
) -> tuple[Path, Path, str, Optional[str]]:
	target_tag, resolved_mode = _resolve_target_tag(
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=binary_mode,
	)
	model_path = models_root / f"transformer_uptoW{int(upto_week)}_{target_tag}.keras"
	history_path = reports_root / f"week_{int(upto_week)}" / f"experiments_history_{target_tag}.json"
	return model_path, history_path, target_tag, resolved_mode


def _expected_static_dim(model_obj) -> Optional[int]:
	if not getattr(model_obj, "with_static_features", False):
		return 0

	try:
		model_inputs = model_obj.inputs
		inputs = model_inputs if isinstance(model_inputs, (list, tuple)) else [model_inputs]
		if len(inputs) >= 4 and inputs[3].shape[-1] is not None:
			return int(inputs[3].shape[-1])
	except Exception:
		pass

	try:
		first_static_layer = model_obj.static_block.layers[0]
		if hasattr(first_static_layer, "kernel"):
			return int(first_static_layer.kernel.shape[0])
	except Exception:
		pass

	return None


def load_npz_split(data_root: Path, split: str, upto_week: int, with_static: bool = True) -> dict:
	fp = data_root / split / f"transformer_uptoW{int(upto_week)}.npz"
	with np.load(fp, allow_pickle=True) as data:
		x_static = data["X_static"].astype(np.float32) if with_static else None
		return {
			"X_seq": data["X_seq"].astype(np.float32),
			"mask_pad": (data["mask_pad"] if "mask_pad" in data.files else data["mask"]).astype(np.int32),
			"mask_activity": (
				data["mask_activity"] if "mask_activity" in data.files else (data["mask_pad"] if "mask_pad" in data.files else data["mask"])
			).astype(np.int32),
			"y": data["y"].astype(np.int64),
			"ids": data["ids"].astype(str),
			"X_static": x_static,
			"static_feature_names": data["static_feature_names"] if "static_feature_names" in data.files else np.array([], dtype=object),
			"static_feature_sources": data["static_feature_sources"] if "static_feature_sources" in data.files else np.array([], dtype=object),
			"activities": data["activities"] if "activities" in data.files else np.array([], dtype=object),
		}


def filter_payload_classes(payload: dict, num_classes: int, binary_mode: Optional[str], paper_baseline: bool) -> dict:
	y = payload["y"]

	if int(num_classes) != 2:
		return payload

	mode = _normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
	if mode == "paper":
		keep = y != 1
		y_bin = np.where(y[keep] == 0, 1, 0).astype(np.int64)
	elif mode == "original":
		keep = y != 0
		y_kept = y[keep]
		y_bin = np.where(y_kept == 1, 1, 0).astype(np.int64)
	else:
		keep = np.ones(len(y), dtype=bool)
		y_bin = np.where(y < 2, 1, 0).astype(np.int64)

	result = {}
	for key, value in payload.items():
		if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(y):
			result[key] = value[keep]
		else:
			result[key] = value
	result["y"] = y_bin
	return result


def load_model_from_history(
	model_path: Path,
	history_path: Path,
	upto_week: int,
	num_classes: int,
	paper_baseline: bool,
	binary_mode: Optional[str],
):
	entries = _load_history_entries(history_path)
	latest_entry = _select_latest_matching_entry(
		entries=entries,
		upto_week=upto_week,
		num_classes=num_classes,
		expected_binary_mode=binary_mode,
		paper_baseline=paper_baseline,
	)
	if latest_entry is None:
		raise ValueError(f"No hay entradas en historial para upto_week={upto_week} con la configuración solicitada")

	model = tf.keras.models.load_model(
		model_path,
		custom_objects=_load_transformer_custom_objects(),
		compile=False,
	)
	return model, latest_entry.get("hyperparameters", {})


def _normalize_shap_array(shap_values_obj, num_classes: int) -> np.ndarray:
	sv = shap_values_obj
	if isinstance(sv, list):
		sv = sv[1] if (int(num_classes) == 2 and len(sv) > 1) else sv[0]
	sv = np.asarray(sv)
	if sv.ndim == 3 and sv.shape[-1] == 1:
		sv = sv[..., 0]
	return sv


def _render_table_png(df: pd.DataFrame, output_path: Path, title: str) -> Path:
	printable = df.copy()
	if "mean_abs_shap" in printable.columns:
		printable["mean_abs_shap"] = printable["mean_abs_shap"].map(lambda value: f"{float(value):.6f}")

	fig_width = max(9, 1.5 * max(len(printable.columns), 4))
	fig_height = max(2.4, 0.55 * (len(printable) + 2))
	fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
	fig.patch.set_facecolor("#111111")
	ax.set_facecolor("#111111")
	ax.axis("off")

	table = ax.table(
		cellText=printable.values,
		colLabels=list(printable.columns),
		cellLoc="center",
		loc="center",
		bbox=[0.02, 0.02, 0.96, 0.9],
	)
	table.auto_set_font_size(False)
	table.set_fontsize(10)

	for (row, col), cell in table.get_celld().items():
		cell.set_edgecolor("#2c2c2c")
		cell.set_linewidth(0.6)
		if row == 0:
			cell.set_facecolor("#2b2b2b")
			cell.set_text_props(color="#f0f0f0", weight="bold")
		else:
			cell.set_facecolor("#1a1a1a" if row % 2 else "#232323")
			cell.set_text_props(color="#f0f0f0")

	ax.set_title(title, fontsize=13, fontweight="bold", color="#f5f5f5", pad=10)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	return output_path


def _render_attention_heatmap_png(matrix: np.ndarray, output_path: Path, title: str) -> Path:
	fig_size = max(5, min(14, matrix.shape[0] * 0.7 + 2))
	fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=180)
	fig.patch.set_facecolor("#111111")
	ax.set_facecolor("#111111")

	im = ax.imshow(matrix, cmap="magma", aspect="auto")
	ax.set_xlabel("Semana atendida", color="#f0f0f0")
	ax.set_ylabel("Semana origen", color="#f0f0f0")
	ax.set_title(title, fontsize=13, fontweight="bold", color="#f5f5f5", pad=10)
	ax.tick_params(colors="#f0f0f0")
	ax.set_xticks(np.arange(matrix.shape[1]))
	ax.set_yticks(np.arange(matrix.shape[0]))
	ax.set_xticklabels([f"W{i+1}" for i in range(matrix.shape[1])], rotation=45, ha="right")
	ax.set_yticklabels([f"W{i+1}" for i in range(matrix.shape[0])])

	colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	colorbar.ax.yaxis.set_tick_params(color="#f0f0f0")
	plt.setp(colorbar.ax.get_yticklabels(), color="#f0f0f0")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	return output_path


def _save_attention_outputs(matrix: np.ndarray, output_dir: Path, base_name: str, title: str) -> tuple[Path, Path]:
	columns = [f"W{i+1:02d}" for i in range(matrix.shape[1])]
	index = [f"W{i+1:02d}" for i in range(matrix.shape[0])]
	df = pd.DataFrame(matrix, index=index, columns=columns)
	csv_path = output_dir / f"{base_name}.csv"
	png_path = output_dir / f"{base_name}.png"
	df.to_csv(csv_path)
	_render_attention_heatmap_png(matrix, png_path, title)
	return csv_path, png_path


def _render_temporal_attention_heatmap_png(df: pd.DataFrame, output_path: Path, title: str) -> Path:
	fig_width = max(14, 1.25 * len(df.columns) + 8)
	fig_height = max(6, 0.24 * len(df.index) + 2)
	fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
	fig.patch.set_facecolor("#111111")
	ax.set_facecolor("#111111")

	display_df = df.copy()
	for column in display_df.columns:
		valid = display_df[column].notna()
		if not valid.any():
			continue
		column_values = display_df.loc[valid, column].astype(float)
		column_min = float(column_values.min())
		column_max = float(column_values.max())
		if column_max > column_min:
			display_df.loc[valid, column] = (column_values - column_min) / (column_max - column_min)
		else:
			display_df.loc[valid, column] = 1.0 if column_max > 0 else 0.0

	mask = df.isna()
	sns.heatmap(
		display_df,
		mask=mask,
		cmap="magma",
		vmin=0.0,
		vmax=1.0,
		annot=df,
		fmt=".3f",
		linewidths=0.5,
		linecolor="#1f1f1f",
		cbar=True,
		cbar_kws={"label": "intensidad relativa dentro de cada semana"},
		ax=ax,
	)

	ax.set_title(title, fontsize=13, fontweight="bold", color="#f5f5f5", pad=12)
	ax.set_xlabel("upto_week del modelo", color="#f0f0f0")
	ax.set_ylabel("semana relativa dentro de la secuencia", color="#f0f0f0")
	ax.tick_params(colors="#f0f0f0")
	plt.setp(ax.get_xticklabels(), rotation=0, color="#f0f0f0")
	plt.setp(ax.get_yticklabels(), rotation=0, color="#f0f0f0")

	if ax.collections and ax.collections[0].colorbar is not None:
		colorbar = ax.collections[0].colorbar
		colorbar.ax.yaxis.set_tick_params(color="#f0f0f0")
		plt.setp(colorbar.ax.get_yticklabels(), color="#f0f0f0")
		colorbar.set_label("intensidad relativa dentro de cada semana", color="#f0f0f0")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	return output_path


def _save_temporal_attention_outputs(df: pd.DataFrame, output_dir: Path, base_name: str, title: str) -> tuple[Path, Path]:
	csv_path = output_dir / f"{base_name}.csv"
	png_path = output_dir / f"{base_name}.png"
	df.to_csv(csv_path)
	_render_temporal_attention_heatmap_png(df, png_path, title)
	return csv_path, png_path


def _render_attention_grid_png(attention_maps: list[dict], output_path: Path, title: str, ncols: int = 4) -> Path:
	if not attention_maps:
		return output_path

	ordered_maps = sorted(attention_maps, key=lambda item: int(item["upto_week"]))
	n_panels = len(ordered_maps)
	ncols = max(1, int(ncols))
	nrows = int(np.ceil(n_panels / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), dpi=180)
	fig.patch.set_facecolor("#111111")
	axes_array = np.atleast_1d(axes).ravel()

	vmin = min(float(np.min(item["matrix"])) for item in ordered_maps)
	vmax = max(float(np.max(item["matrix"])) for item in ordered_maps)
	last_im = None

	for ax, item in zip(axes_array, ordered_maps):
		matrix = np.asarray(item["matrix"], dtype=np.float32)
		ax.set_facecolor("#111111")
		last_im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
		ax.set_title(f"Última capa · upto_week={int(item['upto_week'])}", color="#f5f5f5", fontsize=12, pad=8)
		ax.set_xlabel("Semana key", color="#f0f0f0")
		ax.set_ylabel("Semana query", color="#f0f0f0")
		ax.tick_params(colors="#f0f0f0")

	for ax in axes_array[n_panels:]:
		ax.set_visible(False)

	if last_im is not None:
		fig.colorbar(last_im, ax=axes_array[:n_panels].tolist(), fraction=0.018, pad=0.02)

	fig.suptitle(title, color="#f5f5f5", fontsize=15, fontweight="bold", y=0.995)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.subplots_adjust(left=0.05, right=0.94, bottom=0.06, top=0.92, wspace=0.22, hspace=0.28)
	plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	return output_path


def _save_attention_grid_output(attention_maps: list[dict], output_dir: Path, base_name: str, title: str, ncols: int = 4) -> Optional[Path]:
	if not attention_maps:
		return None
	png_path = output_dir / f"{base_name}.png"
	_render_attention_grid_png(attention_maps, png_path, title, ncols=ncols)
	return png_path


def _aggregate_attention_maps(attention_maps: list[dict]) -> Optional[np.ndarray]:
	if not attention_maps:
		return None

	max_dim = max(int(item["matrix"].shape[0]) for item in attention_maps)
	padded = np.full((len(attention_maps), max_dim, max_dim), np.nan, dtype=np.float32)

	for index, item in enumerate(attention_maps):
		matrix = np.asarray(item["matrix"], dtype=np.float32)
		rows, cols = matrix.shape
		padded[index, :rows, :cols] = matrix

	aggregated = np.nanmean(padded, axis=0)
	return np.nan_to_num(aggregated, nan=0.0)


def _attention_received_by_relative_week(matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(matrix, dtype=np.float32)
	if matrix.size == 0:
		return np.array([], dtype=np.float32)
	return matrix.mean(axis=0).astype(np.float32)


def _build_temporal_attention_heatmap(attention_maps: list[dict]) -> pd.DataFrame:
	if not attention_maps:
		return pd.DataFrame()

	max_week = max(int(item["upto_week"]) for item in attention_maps)
	ordered_weeks = sorted(int(item["upto_week"]) for item in attention_maps)
	column_labels = [str(week) for week in ordered_weeks]
	index_labels = list(range(1, max_week + 1))
	heatmap = pd.DataFrame(np.nan, index=index_labels, columns=column_labels, dtype=float)

	for item in attention_maps:
		upto_week = int(item["upto_week"])
		received = _attention_received_by_relative_week(item["matrix"])
		for week_idx, value in enumerate(received, start=1):
			heatmap.loc[week_idx, str(upto_week)] = float(value)

	for column in heatmap.columns:
		valid = heatmap[column].notna()
		if not valid.any():
			continue
		column_sum = float(heatmap.loc[valid, column].sum())
		if column_sum > 0:
			heatmap.loc[valid, column] = heatmap.loc[valid, column] / column_sum

	return heatmap


def _build_week_cell_summary(df: pd.DataFrame, week_order: list[int]) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()

	summary_df = df.copy()
	summary_df["week_label"] = summary_df["upto_week"].map(lambda value: f"W{int(value):02d}")
	summary_df["cell"] = summary_df.apply(
		lambda row: f"{row['feature']}\n{float(row['mean_abs_shap']):.6f}",
		axis=1,
	)
	wide = summary_df.pivot(index="week_label", columns="rank", values="cell")
	ordered_index = [f"W{int(week):02d}" for week in week_order if f"W{int(week):02d}" in wide.index]
	wide = wide.reindex(index=ordered_index)
	wide = wide.reset_index().rename(columns={"week_label": "week"})
	wide.columns = ["week" if column == "week" else f"top_{int(column)}" for column in wide.columns]
	return wide


def _save_week_outputs(df: pd.DataFrame, output_dir: Path, base_name: str, title: str) -> tuple[Path, Path]:
	csv_path = output_dir / f"{base_name}.csv"
	png_path = output_dir / f"{base_name}.png"
	df.to_csv(csv_path, index=False)
	_render_table_png(df, png_path, title)
	return csv_path, png_path


def _save_global_outputs(df: pd.DataFrame, report_root: Path, long_name: str, wide_name: str, title: str) -> list[Path]:
	outputs: list[Path] = []
	if df.empty:
		return outputs

	long_csv = report_root / f"{long_name}.csv"
	df.to_csv(long_csv, index=False)
	outputs.append(long_csv)

	week_order = sorted(df["upto_week"].unique().tolist())
	wide_df = _build_week_cell_summary(df, week_order)
	if not wide_df.empty:
		wide_csv = report_root / f"{wide_name}.csv"
		wide_png = report_root / f"{wide_name}.png"
		wide_df.to_csv(wide_csv, index=False)
		_render_table_png(wide_df, wide_png, title)
		outputs.extend([wide_csv, wide_png])

	return outputs


def _compute_attention_map(model, x_seq: np.ndarray, mask_activity: np.ndarray, x_static: Optional[np.ndarray] = None) -> np.ndarray:
	x_encoded = model.input_proj(x_seq)
	if getattr(model, "with_static_features", False) and x_static is not None and x_static.shape[1] > 0:
		x_static_emb = model.static_block(x_static, training=False)
		x_static_t = tf.expand_dims(x_static_emb, axis=1)
		x_static_t = tf.broadcast_to(x_static_t, tf.shape(x_encoded))
		fusion_in = tf.concat([x_encoded, x_static_t], axis=-1)
		gate = model.fusion_gate(fusion_in)
		x_encoded = (1.0 - gate) * x_encoded + gate * x_static_t
	x_encoded = model.in_drop(x_encoded, training=False)

	collected = []
	current = x_encoded
	seq_mask = mask_activity

	for encoder in model.encoders:
		attn_mask = encoder.make_attn_mask(seq_mask)
		h = encoder.norm_attn(current)
		_, attention_scores = encoder.mha(
			h,
			h,
			h,
			attention_mask=attn_mask,
			training=False,
			return_attention_scores=True,
		)
		attn_mean = tf.reduce_mean(attention_scores, axis=1)
		collected.append(attn_mean.numpy())

		attn_output = encoder.mha(h, h, h, attention_mask=attn_mask, training=False)
		attn_output = encoder.drop_attn(attn_output, training=False)
		current = current + attn_output
		h_ffn = encoder.norm_ffn(current)
		ffn = encoder.ffn_glu(h_ffn)
		ffn = encoder.ffn_out(ffn)
		ffn = encoder.drop_ffn(ffn, training=False)
		current = current + ffn

	if not collected:
		return np.zeros((x_seq.shape[1], x_seq.shape[1]), dtype=np.float32)

	stacked = np.stack(collected, axis=0)
	return stacked.mean(axis=(0, 1)).astype(np.float32)


def _compute_weekly_shap_importance(
	upto_week: int,
	split: str,
	num_classes: int,
	paper_baseline: bool,
	binary_mode: Optional[str],
	with_static: bool,
	top_k: int,
	shap_bg_size: int,
	shap_explain_size: int,
	shap_nsamples: int,
	rng: np.random.Generator,
	data_root: Path,
	models_root: Path,
	transformer_reports_root: Path,
) -> Optional[dict]:
	model_path, history_path, target_tag, resolved_binary_mode = _week_artifacts(
		upto_week=upto_week,
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=binary_mode,
		models_root=models_root,
		reports_root=transformer_reports_root,
	)
	if not model_path.exists() or not history_path.exists():
		logger.warning(f"⚠️ Saltando W={upto_week}: faltan artefactos de modelo o historial para {target_tag}.")
		return None

	npz_path = data_root / split / f"transformer_uptoW{int(upto_week)}.npz"
	if not npz_path.exists():
		logger.warning(f"⚠️ Saltando W={upto_week}: no existe {npz_path}.")
		return None

	payload_raw = load_npz_split(data_root=data_root, split=split, upto_week=upto_week, with_static=with_static)
	payload = filter_payload_classes(
		payload=payload_raw,
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=resolved_binary_mode,
	)

	x_seq = payload["X_seq"]
	x_static = payload["X_static"]
	mask_activity = payload["mask_activity"]
	activities = payload["activities"]
	static_names = payload["static_feature_names"]

	model, _ = load_model_from_history(
		model_path=model_path,
		history_path=history_path,
		upto_week=upto_week,
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=resolved_binary_mode,
	)

	expected_static_dim = _expected_static_dim(model)
	actual_static_dim = 0 if x_static is None else int(x_static.shape[1])
	if expected_static_dim is not None and expected_static_dim != actual_static_dim:
		logger.warning(
			f"⚠️ Saltando W={upto_week}: el modelo espera {expected_static_dim} features estáticas y el NPZ trae {actual_static_dim}."
		)
		tf.keras.backend.clear_session()
		return None

	n_samples = len(x_seq)
	seq_len = x_seq.shape[1]
	seq_feat_dim = x_seq.shape[2]
	static_dim = 0 if x_static is None else x_static.shape[1]

	def pack_flat(x_seq_local, x_static_local=None):
		x_flat = x_seq_local.reshape(len(x_seq_local), seq_len * seq_feat_dim)
		if x_static_local is not None:
			x_flat = np.concatenate([x_flat, x_static_local], axis=1)
		return x_flat

	def unpack_flat(x_flat):
		x_flat = np.asarray(x_flat)
		x_seq_local = x_flat[:, : seq_len * seq_feat_dim].reshape(-1, seq_len, seq_feat_dim).astype(np.float32)
		x_static_local = x_flat[:, seq_len * seq_feat_dim :].astype(np.float32) if static_dim > 0 else None
		return x_seq_local, x_static_local

	def build_masks(x_seq_local):
		mask_activity_local = (np.abs(x_seq_local).sum(axis=2) > 0).astype(np.int32)
		mask_pad_local = np.ones_like(mask_activity_local, dtype=np.int32)
		return mask_pad_local, mask_activity_local

	def predict_from_flat(x_flat):
		x_seq_local, x_static_local = unpack_flat(x_flat)
		mask_pad_local, mask_activity_local = build_masks(x_seq_local)
		inputs = [x_seq_local, mask_pad_local, mask_activity_local]
		if x_static_local is not None:
			inputs.append(x_static_local)
		probs = model.predict(inputs, verbose=0)
		return probs[:, 1] if int(num_classes) == 2 else np.max(probs, axis=1)

	x_flat = pack_flat(x_seq, x_static)
	bg_size = min(int(shap_bg_size), n_samples)
	explain_size = min(int(shap_explain_size), n_samples)
	bg_idx = rng.choice(n_samples, size=bg_size, replace=False)
	explain_idx = rng.choice(n_samples, size=explain_size, replace=False)

	explainer = shap.KernelExplainer(predict_from_flat, x_flat[bg_idx])
	shap_values = explainer.shap_values(x_flat[explain_idx], nsamples=int(shap_nsamples))
	shap_array = _normalize_shap_array(shap_values, num_classes=num_classes)

	seq_cols = seq_len * seq_feat_dim
	seq_shap = shap_array[:, :seq_cols].reshape(-1, seq_len, seq_feat_dim)
	seq_importance = np.abs(seq_shap).mean(axis=0).mean(axis=0)
	seq_feature_names = [_map_oulad_activity_label(name) for name in activities] if len(activities) == seq_feat_dim else [f"seq_f{i}" for i in range(seq_feat_dim)]
	seq_df = pd.DataFrame(
		{
			"upto_week": int(upto_week),
			"rank": np.arange(1, len(seq_feature_names) + 1),
			"feature": seq_feature_names,
			"mean_abs_shap": seq_importance,
		}
	).sort_values("mean_abs_shap", ascending=False).head(int(top_k)).reset_index(drop=True)
	seq_df["rank"] = np.arange(1, len(seq_df) + 1)

	if static_dim > 0:
		static_feature_names = [_map_static_feature_label(name) for name in static_names] if len(static_names) == static_dim else [f"static_f{i}" for i in range(static_dim)]
		static_importance = np.abs(shap_array[:, seq_cols:]).mean(axis=0)
		static_df = pd.DataFrame(
			{
				"upto_week": int(upto_week),
				"rank": np.arange(1, len(static_feature_names) + 1),
				"feature": static_feature_names,
				"mean_abs_shap": static_importance,
			}
		).sort_values("mean_abs_shap", ascending=False).head(int(top_k)).reset_index(drop=True)
		static_df["rank"] = np.arange(1, len(static_df) + 1)
	else:
		static_df = pd.DataFrame(columns=["upto_week", "rank", "feature", "mean_abs_shap"])

	attention_size = min(int(shap_explain_size), n_samples)
	attention_idx = rng.choice(n_samples, size=attention_size, replace=False)
	attention_map = _compute_attention_map(
		model,
		x_seq=x_seq[attention_idx],
		mask_activity=mask_activity[attention_idx],
		x_static=x_static[attention_idx] if x_static is not None else None,
	)

	tf.keras.backend.clear_session()
	return {
		"upto_week": int(upto_week),
		"target_tag": target_tag,
		"seq_df": seq_df,
		"static_df": static_df,
		"attention_map": attention_map,
	}


def generate_xai_reports(
	weeks_csv: Optional[str] = None,
	split: str = XAI_PARAMS.split,
	top_k: int = XAI_PARAMS.top_k,
	shap_bg_size: int = XAI_PARAMS.shap_bg_size,
	shap_explain_size: int = XAI_PARAMS.shap_explain_size,
	shap_nsamples: int = XAI_PARAMS.shap_nsamples,
	seed: int = XAI_PARAMS.seed,
	with_static: bool = XAI_PARAMS.with_static,
	num_classes: Optional[int] = None,
	paper_baseline: Optional[bool] = None,
	binary_mode: Optional[str] = None,
	data_root: Path = TRANSFORMER_FEATURES_ROOT,
	models_root: Path = TRANSFORMER_MODELS_ROOT,
	reports_root: Path = XAI_REPORTS_ROOT,
	transformer_reports_root: Path = TRANSFORMER_REPORTS_ROOT,
) -> dict:
	transformer_params = _load_active_transformer_params()
	resolved_num_classes = int(num_classes if num_classes is not None else getattr(transformer_params, "num_classes", 2))
	resolved_paper_baseline = bool(
		paper_baseline if paper_baseline is not None else getattr(transformer_params, "paper_baseline", True)
	)
	resolved_binary_mode = binary_mode if binary_mode is not None else getattr(transformer_params, "binary_mode", None)
	target_tag, resolved_binary_mode = _resolve_target_tag(
		num_classes=resolved_num_classes,
		paper_baseline=resolved_paper_baseline,
		binary_mode=resolved_binary_mode,
	)
	reports_root.mkdir(parents=True, exist_ok=True)
	_migrate_legacy_xai_reports(reports_root)
	resolved_xai_reports_root = resolve_report_dir(
		reports_root,
		num_classes=resolved_num_classes,
		paper_baseline=resolved_paper_baseline,
		binary_mode=resolved_binary_mode,
	)
	migrate_legacy_transformer_reports(transformer_reports_root)
	resolved_transformer_reports_root = resolve_report_dir(
		transformer_reports_root,
		num_classes=resolved_num_classes,
		paper_baseline=resolved_paper_baseline,
		binary_mode=resolved_binary_mode,
	)

	weeks = _resolve_weeks(weeks_csv=weeks_csv, split=split, data_root=data_root)
	rng = np.random.default_rng(int(seed))
	resolved_xai_reports_root.mkdir(parents=True, exist_ok=True)
	_cleanup_residual_xai_files(report_root=resolved_xai_reports_root, legacy_root=reports_root)

	seq_frames = []
	static_frames = []
	attention_outputs = []
	attention_maps = []

	for upto_week in weeks:
		week_result = _compute_weekly_shap_importance(
			upto_week=upto_week,
			split=split,
			num_classes=resolved_num_classes,
			paper_baseline=resolved_paper_baseline,
			binary_mode=resolved_binary_mode,
			with_static=with_static,
			top_k=top_k,
			shap_bg_size=shap_bg_size,
			shap_explain_size=shap_explain_size,
			shap_nsamples=shap_nsamples,
			rng=rng,
			data_root=data_root,
			models_root=models_root,
			transformer_reports_root=resolved_transformer_reports_root,
		)
		if week_result is None:
			continue

		week_dir = resolved_xai_reports_root / f"week_{int(upto_week)}"
		week_dir.mkdir(parents=True, exist_ok=True)
		seq_df = week_result["seq_df"]
		static_df = week_result["static_df"]
		attention_map = week_result["attention_map"]

		_save_week_outputs(
			seq_df,
			week_dir,
			f"xai_top_sequential_{target_tag}",
			f"Top SHAP Secuenciales | W{int(upto_week):02d} | {target_tag}",
		)
		seq_frames.append(seq_df)

		if not static_df.empty:
			_save_week_outputs(
				static_df,
				week_dir,
				f"xai_top_static_{target_tag}",
				f"Top SHAP Estáticas | W{int(upto_week):02d} | {target_tag}",
			)
			static_frames.append(static_df)

		attention_outputs.extend(
			_save_attention_outputs(
				attention_map,
				week_dir,
				f"attention_map_{target_tag}",
				f"Mapa de Atención | W{int(upto_week):02d} | {target_tag}",
			)
		)
		attention_maps.append({"upto_week": int(upto_week), "matrix": attention_map})

	seq_global = pd.concat(seq_frames, ignore_index=True) if seq_frames else pd.DataFrame(columns=["upto_week", "rank", "feature", "mean_abs_shap"])
	static_global = pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame(columns=["upto_week", "rank", "feature", "mean_abs_shap"])

	seq_outputs = _save_global_outputs(
		seq_global,
		resolved_xai_reports_root,
		f"xai_global_top_sequential_long_{target_tag}",
		f"xai_global_top_sequential_wide_{target_tag}",
		f"Top SHAP Secuenciales por Semana | {target_tag}",
	)
	static_outputs = _save_global_outputs(
		static_global,
		resolved_xai_reports_root,
		f"xai_global_top_static_long_{target_tag}",
		f"xai_global_top_static_wide_{target_tag}",
		f"Top SHAP Estáticas por Semana | {target_tag}",
	)

	global_attention_outputs = []
	if attention_maps:
		temporal_attention_heatmap = _build_temporal_attention_heatmap(attention_maps)
		if not temporal_attention_heatmap.empty:
			global_attention_outputs.extend(
				_save_temporal_attention_outputs(
					temporal_attention_heatmap,
					resolved_xai_reports_root,
					f"attention_temporal_heatmap_{target_tag}",
					f"Atención media recibida por semana relativa y ventana | {target_tag}",
				)
			)

		attention_grid_output = _save_attention_grid_output(
			attention_maps,
			resolved_xai_reports_root,
			f"attention_last_layer_grid_{target_tag}",
			f"Rejilla de mapas de atención por ventana | {target_tag}",
			ncols=4,
		)
		if attention_grid_output is not None:
			global_attention_outputs.append(attention_grid_output)

		aggregated_attention_map = _aggregate_attention_maps(attention_maps)
		global_attention_outputs.extend(
			_save_attention_outputs(
				aggregated_attention_map,
				resolved_xai_reports_root,
				f"attention_global_mean_{target_tag}",
				f"Mapa de Atención Global Medio | {target_tag}",
			)
		)
		attention_summary = pd.DataFrame(
			{
				"upto_week": [item["upto_week"] for item in attention_maps],
				"max_attention": [float(np.max(item["matrix"])) for item in attention_maps],
				"mean_attention": [float(np.mean(item["matrix"])) for item in attention_maps],
			}
		)
		global_attention_outputs.extend(
			_save_week_outputs(
				attention_summary,
				resolved_xai_reports_root,
				f"attention_global_summary_{target_tag}",
				f"Resumen Global de Atención | {target_tag}",
			)
		)

	logger.info(f"✅ XAI SHAP completado para {target_tag} | semanas procesadas: {sorted(seq_global['upto_week'].unique().tolist()) if not seq_global.empty else []}")
	return {
		"target_tag": target_tag,
		"weeks": weeks,
		"seq_global": seq_global,
		"static_global": static_global,
		"outputs": seq_outputs + static_outputs + attention_outputs + global_attention_outputs,
	}


@app.command("run-all")
def run_all(
	weeks_csv: Optional[str] = typer.Option(XAI_PARAMS.weeks_csv, help="Semanas separadas por coma. Si no se indica, usa W_WINDOWS."),
	split: str = typer.Option(XAI_PARAMS.split, help="Split sobre el que calcular XAI."),
	top_k: int = typer.Option(XAI_PARAMS.top_k, help="Número de features top a guardar por semana."),
	shap_bg_size: int = typer.Option(XAI_PARAMS.shap_bg_size, help="Tamaño del background set para SHAP."),
	shap_explain_size: int = typer.Option(XAI_PARAMS.shap_explain_size, help="Número de muestras explicadas por semana."),
	shap_nsamples: int = typer.Option(XAI_PARAMS.shap_nsamples, help="Número de evaluaciones aproximadas por explicación SHAP."),
	seed: int = typer.Option(XAI_PARAMS.seed, help="Semilla aleatoria para muestreo reproducible."),
	with_static: bool = typer.Option(XAI_PARAMS.with_static, help="Usar o no features estáticas desde el NPZ."),
	num_classes: Optional[int] = typer.Option(None, help="Sobrescribe num_classes activo."),
	paper_baseline: Optional[bool] = typer.Option(None, help="Sobrescribe paper_baseline activo."),
	binary_mode: Optional[str] = typer.Option(None, help="Sobrescribe binary_mode activo."),
):
	"""Genera salidas XAI SHAP por semana y consolidado global para la configuración activa del transformer."""
	result = generate_xai_reports(
		weeks_csv=weeks_csv,
		split=split,
		top_k=top_k,
		shap_bg_size=shap_bg_size,
		shap_explain_size=shap_explain_size,
		shap_nsamples=shap_nsamples,
		seed=seed,
		with_static=with_static,
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=binary_mode,
	)
	if result["seq_global"].empty and result["static_global"].empty:
		raise typer.Exit(code=1)


if __name__ == "__main__":
	app()
