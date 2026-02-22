import json
import pandas as pd
from pathlib import Path
import sys

def find_best_cases(file_path, top_n=10, sort_by="val_auc"):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: El archivo {file_path} no existe.")
        return

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("returncode") == 0 and record.get("metrics"):
                    # Aplanamos el JSON para que pandas lo lea fácil
                    entry = {
                        "cfg_id": record["cfg_id"],
                        **record["cfg"],
                        **record["metrics"]
                    }
                    data.append(entry)
            except Exception as e:
                continue

    if not data:
        print("No se encontraron resultados válidos en el archivo.")
        return

    df = pd.DataFrame(data)
    
    # Asegurarnos de que la métrica de ordenación existe
    if sort_by not in df.columns:
        available_metrics = [c for c in df.columns if c.startswith("val_")]
        print(f"Métrica '{sort_by}' no encontrada. Métricas disponibles: {available_metrics}")
        sort_by = "val_balanced_acc" if "val_balanced_acc" in df.columns else available_metrics[0]
        print(f"Usando {sort_by} por defecto.")

    # Ordenar (descendente para métricas donde 'más es mejor')
    ascending = False
    if "loss" in sort_by:
        ascending = True
        
    df_sorted = df.sort_values(by=sort_by, ascending=ascending).head(top_n)

    print(f"\n TOP {top_n} CASOS (Ordenados por {sort_by}):")
    print("=" * 100)
    
    # Columnas a mostrar (Hparams + Métricas clave)
    hparams = list(record["cfg"].keys())
    metrics = ["val_balanced_acc", "val_f1", "val_auc", "val_accuracy"]
    
    cols_to_show = ["cfg_id"] + hparams + metrics
    # Filtrar solo las que existen
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    print(df_sorted[cols_to_show].to_string(index=False))
    
    best = df_sorted.iloc[0]
    print("\n" + "=" * 100)
    print(f" EL MEJOR CASO (ID: {best['cfg_id']}):")
    print("-" * 50)
    print("Hyperparameters:")
    for hp in hparams:
        print(f"  - {hp}: {best[hp]}")
    print("\nMetrics:")
    for m in metrics:
        if m in best:
            print(f"  - {m}: {best[m]:.4f}")
    print("=" * 100)

    # Exportar CSV con cfg_id y todas las métricas
    metric_cols = [c for c in df.columns if c.startswith("val_")]
    csv_cols = ["cfg_id"] + metric_cols
    csv_path = path.parent / "metrics_summary.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"\n✅ Resumen de métricas guardado en: {csv_path}")

if __name__ == "__main__":
    file_path = "/workspace/TFM_education_ai_analytics/reports/hparam_search/results.jsonl"
    metric = "val_auc"
    if len(sys.argv) > 1:
        metric = sys.argv[1]
    
    find_best_cases(file_path, sort_by=metric)
