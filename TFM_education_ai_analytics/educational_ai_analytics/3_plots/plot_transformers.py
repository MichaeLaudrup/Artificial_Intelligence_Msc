import importlib
import json
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from educational_ai_analytics.config import DATA_DIR, REPORTS_DIR

from .style import set_style

app = typer.Typer(help="Visualizaciones para Transformers.")
set_style()

TRANSFORMER_REPORTS_ROOT = REPORTS_DIR / "transformer_training"
TRANSFORMER_FEATURES_ROOT = DATA_DIR / "6_transformer_features"
SUMMARY_COLUMNS = [
    "upto_week",
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "auc",
]


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Genera resúmenes globales del transformer."""
    if ctx.invoked_subcommand is None:
        validation_result = generate_global_split_summary(split_name="validation")
        test_result = generate_global_split_summary(split_name="test")
        if validation_result is None and test_result is None:
            raise typer.Exit(code=1)


def _load_active_transformer_params():
    try:
        module = importlib.import_module("educational_ai_analytics.2_modeling.transformers.hyperparams")
        return getattr(module, "TRANSFORMER_PARAMS", None)
    except Exception as exc:
        logger.warning(f"No se pudieron cargar los hyperparams activos del transformer: {exc}")
        return None


def _normalize_binary_mode(paper_baseline: bool, binary_mode: Optional[str]) -> str:
    if binary_mode is None:
        return "paper" if paper_baseline else "original"

    mode = str(binary_mode).strip().lower()
    aliases = {
        "paper": "paper",
        "baseline": "paper",
        "original": "original",
        "success_vs_risk": "success_vs_risk",
        "risk": "success_vs_risk",
        "passdist_vs_failwithdraw": "success_vs_risk",
    }
    if mode not in aliases:
        raise ValueError(
            f"binary_mode inválido: {binary_mode}. Usa uno de: paper|baseline|original|success_vs_risk"
        )
    return aliases[mode]


def _resolve_target_tag(num_classes: int, paper_baseline: bool, binary_mode: Optional[str]) -> tuple[str, Optional[str]]:
    if int(num_classes) == 2:
        resolved_mode = _normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
        return f"{num_classes}clases_{resolved_mode}", resolved_mode
    return f"{num_classes}clases", None


def _resolve_history_filename(
    num_classes: int,
    paper_baseline: bool,
    binary_mode: Optional[str],
    history_filename: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    target_tag, resolved_mode = _resolve_target_tag(
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
    )
    filename = history_filename or f"experiments_history_{target_tag}.json"
    return filename, target_tag, resolved_mode


def _parse_week_token(token: str) -> Optional[int]:
    token = str(token).strip()
    if not token:
        return None
    if token.startswith("week_"):
        token = token.removeprefix("week_")
    if token.startswith("transformer_uptoW"):
        token = token.removeprefix("transformer_uptoW")
    return int(token) if token.isdigit() else None


def _discover_candidate_weeks(
    report_root: Path,
    data_root: Path,
    preferred_weeks: Optional[Iterable[int]] = None,
) -> list[int]:
    weeks = {int(week) for week in (preferred_weeks or [])}

    for path in report_root.glob("week_*"):
        parsed = _parse_week_token(path.name)
        if parsed is not None:
            weeks.add(parsed)

    validation_dir = data_root / "validation"
    if validation_dir.exists():
        for path in validation_dir.glob("transformer_uptoW*.npz"):
            parsed = _parse_week_token(path.stem)
            if parsed is not None:
                weeks.add(parsed)

    return sorted(weeks)


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


def _filtered_sample_count(
    y: np.ndarray,
    num_classes: int,
    paper_baseline: bool,
    binary_mode: Optional[str],
) -> int:
    if int(num_classes) != 2:
        return int(len(y))

    mode = _normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
    if mode == "paper":
        return int(np.count_nonzero(y != 1))
    if mode == "original":
        return int(np.count_nonzero(y != 0))
    return int(len(y))


def _load_split_sample_count(
    data_root: Path,
    split_name: str,
    upto_week: int,
    num_classes: int,
    paper_baseline: bool,
    binary_mode: Optional[str],
) -> Optional[int]:
    npz_path = data_root / split_name / f"transformer_uptoW{int(upto_week)}.npz"
    if not npz_path.exists():
        return None

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            y = np.asarray(data["y"], dtype=np.int64)
    except Exception as exc:
        logger.warning(f"No se pudo leer {npz_path} para contar muestras de {split_name}: {exc}")
        return None

    return _filtered_sample_count(
        y=y,
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
    )


def _metric_or_none(metrics: dict, key: str) -> Optional[float]:
    value = metrics.get(key)
    return None if value is None else float(value)


def build_split_summary_dataframe(
    report_root: Path = TRANSFORMER_REPORTS_ROOT,
    data_root: Path = TRANSFORMER_FEATURES_ROOT,
    split_name: str = "validation",
    weeks: Optional[Iterable[int]] = None,
    num_classes: Optional[int] = None,
    paper_baseline: Optional[bool] = None,
    binary_mode: Optional[str] = None,
    history_filename: Optional[str] = None,
) -> pd.DataFrame:
    params = _load_active_transformer_params()

    resolved_num_classes = int(num_classes if num_classes is not None else getattr(params, "num_classes", 2))
    resolved_paper_baseline = bool(
        paper_baseline if paper_baseline is not None else getattr(params, "paper_baseline", True)
    )
    resolved_binary_mode = binary_mode if binary_mode is not None else getattr(params, "binary_mode", None)

    history_file, _, expected_binary_mode = _resolve_history_filename(
        num_classes=resolved_num_classes,
        paper_baseline=resolved_paper_baseline,
        binary_mode=resolved_binary_mode,
        history_filename=history_filename,
    )

    rows: list[dict] = []
    for upto_week in _discover_candidate_weeks(report_root=report_root, data_root=data_root, preferred_weeks=weeks):
        history_path = report_root / f"week_{int(upto_week)}" / history_file
        entries = _load_history_entries(history_path)
        if not entries:
            continue

        latest_entry = _select_latest_matching_entry(
            entries=entries,
            upto_week=upto_week,
            num_classes=resolved_num_classes,
            expected_binary_mode=expected_binary_mode,
            paper_baseline=resolved_paper_baseline,
        )
        if latest_entry is None:
            continue

        metrics_key = f"{split_name}_metrics"
        split_metrics = latest_entry.get(metrics_key, {})
        if not split_metrics:
            continue

        rows.append({
            "upto_week": int(upto_week),
            "n_samples": _load_split_sample_count(
                data_root=data_root,
                split_name=split_name,
                upto_week=upto_week,
                num_classes=resolved_num_classes,
                paper_baseline=resolved_paper_baseline,
                binary_mode=expected_binary_mode,
            ),
            "accuracy": _metric_or_none(split_metrics, "accuracy"),
            "balanced_accuracy": _metric_or_none(split_metrics, "balanced_accuracy"),
            "precision": _metric_or_none(split_metrics, "precision"),
            "recall": _metric_or_none(split_metrics, "recall"),
            "auc": _metric_or_none(split_metrics, "auc_ovr"),
        })

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary_df = pd.DataFrame(rows).sort_values("upto_week").reset_index(drop=True)
    return summary_df[SUMMARY_COLUMNS]


def _format_summary_for_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["upto_week", "n_samples"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{int(value)}")

    for column in ["accuracy", "balanced_accuracy", "precision", "recall", "auc"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return formatted


def render_summary_table_png(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    formatted = _format_summary_for_table(summary_df)
    n_rows = max(len(formatted), 1)
    n_cols = max(len(formatted.columns), 1)

    fig_width = max(12, n_cols * 1.75)
    fig_height = max(2.6, 1.2 + n_rows * 0.48)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.axis("off")

    table = ax.table(
        cellText=formatted.values,
        colLabels=list(formatted.columns),
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2c2c2c")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="#f0f0f0", weight="bold")
        else:
            cell.set_facecolor("#1a1a1a" if row % 2 else "#232323")
            cell.set_text_props(color="#f0f0f0")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#f5f5f5", pad=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def generate_global_split_summary(
    report_root: Path = TRANSFORMER_REPORTS_ROOT,
    data_root: Path = TRANSFORMER_FEATURES_ROOT,
    split_name: str = "validation",
    weeks: Optional[Iterable[int]] = None,
    num_classes: Optional[int] = None,
    paper_baseline: Optional[bool] = None,
    binary_mode: Optional[str] = None,
    history_filename: Optional[str] = None,
) -> Optional[tuple[pd.DataFrame, Path, Path]]:
    params = _load_active_transformer_params()
    resolved_num_classes = int(num_classes if num_classes is not None else getattr(params, "num_classes", 2))
    resolved_paper_baseline = bool(
        paper_baseline if paper_baseline is not None else getattr(params, "paper_baseline", True)
    )
    resolved_binary_mode = binary_mode if binary_mode is not None else getattr(params, "binary_mode", None)

    _, target_tag, _ = _resolve_history_filename(
        num_classes=resolved_num_classes,
        paper_baseline=resolved_paper_baseline,
        binary_mode=resolved_binary_mode,
        history_filename=history_filename,
    )

    summary_df = build_split_summary_dataframe(
        report_root=report_root,
        data_root=data_root,
        split_name=split_name,
        weeks=weeks,
        num_classes=resolved_num_classes,
        paper_baseline=resolved_paper_baseline,
        binary_mode=resolved_binary_mode,
        history_filename=history_filename,
    )
    if summary_df.empty:
        logger.warning(
            f"No se encontraron experimentos compatibles para generar el resumen global de {split_name} ({target_tag})."
        )
        return None

    csv_path = report_root / f"{split_name}_summary_{target_tag}.csv"
    png_path = report_root / f"{split_name}_summary_{target_tag}.png"
    summary_df.to_csv(csv_path, index=False)
    render_summary_table_png(
        summary_df=summary_df,
        output_path=png_path,
        title=f"Transformer {split_name.capitalize()} Summary | {target_tag}",
    )

    logger.info(f"✅ CSV global guardado en: {csv_path}")
    logger.info(f"✅ Tabla PNG global guardada en: {png_path}")
    return summary_df, csv_path, png_path


def generate_global_validation_summary(
    report_root: Path = TRANSFORMER_REPORTS_ROOT,
    data_root: Path = TRANSFORMER_FEATURES_ROOT,
    weeks: Optional[Iterable[int]] = None,
    num_classes: Optional[int] = None,
    paper_baseline: Optional[bool] = None,
    binary_mode: Optional[str] = None,
    history_filename: Optional[str] = None,
) -> Optional[tuple[pd.DataFrame, Path, Path]]:
    return generate_global_split_summary(
        report_root=report_root,
        data_root=data_root,
        split_name="validation",
        weeks=weeks,
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
        history_filename=history_filename,
    )


def generate_global_test_summary(
    report_root: Path = TRANSFORMER_REPORTS_ROOT,
    data_root: Path = TRANSFORMER_FEATURES_ROOT,
    weeks: Optional[Iterable[int]] = None,
    num_classes: Optional[int] = None,
    paper_baseline: Optional[bool] = None,
    binary_mode: Optional[str] = None,
    history_filename: Optional[str] = None,
) -> Optional[tuple[pd.DataFrame, Path, Path]]:
    return generate_global_split_summary(
        report_root=report_root,
        data_root=data_root,
        split_name="test",
        weeks=weeks,
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
        history_filename=history_filename,
    )


@app.command("validation-summary")
def validation_summary(
    report_root: Path = typer.Option(TRANSFORMER_REPORTS_ROOT, help="Raíz de reports/transformer_training."),
    data_root: Path = typer.Option(TRANSFORMER_FEATURES_ROOT, help="Raíz de data/6_transformer_features."),
    num_classes: Optional[int] = typer.Option(None, help="Sobrescribe num_classes activo."),
    paper_baseline: Optional[bool] = typer.Option(None, help="Sobrescribe paper_baseline activo."),
    binary_mode: Optional[str] = typer.Option(None, help="Sobrescribe binary_mode activo."),
    history_filename: Optional[str] = typer.Option(None, help="Nombre del history JSON a consolidar."),
):
    """Genera un CSV y una tabla PNG global con métricas de validación del transformer."""
    result = generate_global_validation_summary(
        report_root=report_root,
        data_root=data_root,
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
        history_filename=history_filename,
    )
    if result is None:
        raise typer.Exit(code=1)


@app.command("test-summary")
def test_summary(
    report_root: Path = typer.Option(TRANSFORMER_REPORTS_ROOT, help="Raíz de reports/transformer_training."),
    data_root: Path = typer.Option(TRANSFORMER_FEATURES_ROOT, help="Raíz de data/6_transformer_features."),
    num_classes: Optional[int] = typer.Option(None, help="Sobrescribe num_classes activo."),
    paper_baseline: Optional[bool] = typer.Option(None, help="Sobrescribe paper_baseline activo."),
    binary_mode: Optional[str] = typer.Option(None, help="Sobrescribe binary_mode activo."),
    history_filename: Optional[str] = typer.Option(None, help="Nombre del history JSON a consolidar."),
):
    """Genera un CSV y una tabla PNG global con métricas de test del transformer."""
    result = generate_global_test_summary(
        report_root=report_root,
        data_root=data_root,
        num_classes=num_classes,
        paper_baseline=paper_baseline,
        binary_mode=binary_mode,
        history_filename=history_filename,
    )
    if result is None:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
