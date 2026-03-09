import shutil
from pathlib import Path
from typing import Optional


def normalize_binary_mode(paper_baseline: bool, binary_mode: Optional[str]) -> str:
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


def resolve_report_scope_name(num_classes: int, paper_baseline: bool, binary_mode: Optional[str]) -> str:
	if int(num_classes) == 2:
		resolved_mode = normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
		return f"binary_classification_{resolved_mode}"
	if int(num_classes) == 3:
		return "ternary_classification"
	if int(num_classes) == 4:
		return "quaternary_classification"
	return f"{int(num_classes)}_class_classification"


def resolve_report_dir(report_root: Path, num_classes: int, paper_baseline: bool, binary_mode: Optional[str]) -> Path:
	return Path(report_root) / resolve_report_scope_name(
		num_classes=num_classes,
		paper_baseline=paper_baseline,
		binary_mode=binary_mode,
	)


def infer_report_scope_from_name(name: str) -> Optional[str]:
	name = str(name)
	if "_2clases_paper" in name:
		return "binary_classification_paper"
	if "_2clases_success_vs_risk" in name:
		return "binary_classification_success_vs_risk"
	if "_2clases_original" in name:
		return "binary_classification_original"
	if "_3clases" in name:
		return "ternary_classification"
	if "_4clases" in name:
		return "quaternary_classification"
	if "_2clases" in name or name == "experiments_history.json":
		return "binary_classification_legacy"
	return None


def migrate_legacy_transformer_reports(report_root: Path) -> list[str]:
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