siimport os
from pathlib import Path

base_dir = Path("/workspace/TFM_education_ai_analytics/educational_ai_analytics/modeling")

files_to_delete = ["train.py", "encode.py", "predict.py"]

for f in files_to_delete:
    p = base_dir / f
    if p.exists():
        print(f"Borrando archivo antiguo: {f}")
        os.remove(p)

print("🧹 Limpieza de scripts antiguos completada.")
