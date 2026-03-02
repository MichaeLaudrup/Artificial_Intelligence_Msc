import json
from pathlib import Path


def _preferred_metric_order(num_classes: int) -> list[str]:
    if num_classes == 2:
        return [
            "loss",
            "accuracy",
            "balanced_accuracy",
            "auc_ovr",
            "precision",
            "recall",
            "f1_score",
        ]
    base_multiclass = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "auc_ovr",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    if num_classes == 4:
        return base_multiclass + ["top_2_acc"]
    return base_multiclass


def _ordered_metrics_to_show(metrics_a: dict, metrics_b: dict, num_classes: int) -> list[str]:
    preferred = _preferred_metric_order(num_classes)
    all_metrics = set(metrics_a.keys()).union(set(metrics_b.keys()))
    ordered = [k for k in preferred if k in all_metrics]
    if num_classes == 2:
        # En binario sí mostramos extras para mantener compatibilidad histórica.
        ordered += sorted(list(all_metrics - set(ordered)))
    return ordered


def _print_metrics_snapshot(exp: dict, index_label: str = "1") -> None:
    t = exp.get("timestamp", f"Exp {index_label}")
    metrics = exp.get("validation_metrics", {})
    num_classes = int(exp.get("hyperparameters", {}).get("num_classes", 2))

    print(f"\n📊 MÉTRICAS ACTUALES (sin comparativa): [{index_label}] {t}")
    if not metrics:
        print("  ⚠️  No hay métricas de validación registradas en esta corrida.")
        return

    print("\n  📈 MÉTRICAS (Validación):")
    for k in _ordered_metrics_to_show(metrics, {}, num_classes):
        v = metrics.get(k)
        if v is None:
            print(f"      - {k.ljust(18)}: [No disponible]")
        else:
            print(f"      - {k.ljust(18)}: {float(v):.4f}")


def compare_experiments(history_path: Path = Path("reports/transformer_training/experiments_history.json")):
    if isinstance(history_path, str):
        history_path = Path(history_path)
    if not history_path.exists():
        print("No hay historial de experimentos para comparar.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print("El historial está corrupto.")
            return

    if len(history) < 2:
        print("Solo hay 1 experimento en el historial; no hay corrida anterior para comparar.")
        _print_metrics_snapshot(history[-1], index_label=str(len(history)))
        return

    print(f"\n📊 Análisis de Último Experimento ({len(history)} corridas totales en el historial)\n" + "="*80)

    # Mostrar solo la comparativa más reciente (el último contra el penúltimo)
    for i in range(len(history) - 1, len(history) - 2, -1):
        exp1 = history[i - 1]
        exp2 = history[i]

        t1 = exp1.get("timestamp", f"Exp {i-1}")
        t2 = exp2.get("timestamp", f"Exp {i}")

        print(f"\n🚀 COMPARATIVA: [{i}] {t1}  vs  [{i+1}] {t2}")

        # Comparar hiperparámetros
        params1 = exp1.get("hyperparameters", {})
        params2 = exp2.get("hyperparameters", {})

        changed_params = {}
        # Unir claves de ambos diccionarios para no dejarnos ninguno
        all_keys = set(params1.keys()).union(set(params2.keys()))
        for k in all_keys:
            v1 = params1.get(k)
            v2 = params2.get(k)
            if v1 != v2:
                changed_params[k] = (v1, v2)

        if not changed_params:
            print("  ⚙️  No hubo cambios en los hiperparámetros.")
        else:
            print("  ⚙️  HIPERPARÁMETROS MODIFICADOS:")
            for k, (v1, v2) in changed_params.items():
                print(f"      * {k}:  {v1}  --->  {v2}")

        # Comparar métricas
        metrics1 = exp1.get("validation_metrics", {})
        metrics2 = exp2.get("validation_metrics", {})
        num_classes_exp2 = int(exp2.get("hyperparameters", {}).get("num_classes", 2))
        
        print("\n  📈 EVOLUCIÓN DE MÉTRICAS (Validación):")
        metrics_to_show = _ordered_metrics_to_show(metrics1, metrics2, num_classes_exp2)
        for k in metrics_to_show:
            m1 = metrics1.get(k)
            m2 = metrics2.get(k)
            
            if m1 is None and m2 is not None:
                print(f"      - {k.ljust(18)}: [No Exisite] ->  {m2:.4f}  (🌟 NUEVA)")
                continue
            if m1 is None or m2 is None:
                continue
            
            diff = m2 - m1
            pct_change = (diff / abs(m1)) * 100 if m1 != 0 else 0
            
            # Dirección de mejora
            if "loss" in k.lower():
                is_better = diff < 0
            else:
                is_better = diff > 0
                
            if abs(pct_change) < 0.05:
                # Margen de ruido
                emoji = "⬜ Estable"
            elif is_better:
                emoji = "🟩 MEJORÓ"
            else:
                emoji = "🟥 empeoró"
                
            print(f"      - {k.ljust(18)}: {m1:.4f}  ->  {m2:.4f}  (Δ: {diff:+.4f} | {pct_change:+.2f}%)  {emoji}")
            
        print("-" * 80)

if __name__ == "__main__":
    import sys
    
    reports_dir = Path("reports/transformer_training")
    
    if len(sys.argv) > 1:
        # If user provides a specific file path
        compare_experiments(Path(sys.argv[1]))
    else:
        # Search for all metric files in week_* folders
        history_files = sorted(list(reports_dir.glob("week_*/experiments_history.json")))
        
        if not history_files:
            print(f"No se encontraron historiales en {reports_dir}/week_*/")
        else:
            for hist in history_files:
                week_match = hist.parent.name
                print(f"\n" + "*"*80)
                print(f"🌟 Analizando historial de la carpeta: {week_match.upper()}")
                compare_experiments(history_path=hist)
