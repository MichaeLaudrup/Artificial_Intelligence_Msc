import json
from pathlib import Path

def compare_experiments():
    history_path = Path("reports/transformer_training/experiments_history.json")
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
        print("Se necesitan al menos 2 experimentos en el historial para hacer comparaciones.")
        return

    print(f"\n📊 Análisis de Historial de Experimentos ({len(history)} corridas totales)\n" + "="*80)

    for i in range(1, len(history)):
        exp1 = history[i-1]
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
            # Para listas, asegurarnos de que se comparan correctamente
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
        
        print("\n  📈 EVOLUCIÓN DE MÉTRICAS (Validación):")
        for k in metrics1.keys():
            m1 = metrics1.get(k)
            m2 = metrics2.get(k)
            if m1 is None or m2 is None: continue
            
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
    compare_experiments()
