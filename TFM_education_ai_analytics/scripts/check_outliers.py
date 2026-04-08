import pandas as pd
import numpy as np

# 1. Cargar interacciones
print("Cargando interactions.csv...")
df = pd.read_csv("data/1_interim/interactions.csv")

# 2. Filtrar pre-start (día < 0) como hace el builder
pre = df[df["date"] < 0].copy()

# 3. Agrupar por estudiante (combinando id_student y curso para ser exactos)
pre["unique_id"] = pre["id_student"].astype(str) + "_" + pre["code_module"] + "_" + pre["code_presentation"]
stats = pre.groupby("unique_id")["sum_click"].sum().sort_values(ascending=False)

print(f"\nTotal estudiantes con actividad pre-start: {len(stats)}")
print("-" * 50)
print("TOP 10 ESTUDIANTES (OUTLIERS POTENCIALES):")
print(stats.head(10))

# 4. Calcular percentiles clave
p99 = stats.quantile(0.99)
p995 = stats.quantile(0.995)
p999 = stats.quantile(0.999)
max_val = stats.max()

print("\nDISTRIBUCIÓN DE CLICKS (Pre-start):")
print(f"Percentil 99.0 : {p99:>8.1f}")
print(f"Percentil 99.5 : {p995:>8.1f}  <-- Corte actual")
print(f"Percentil 99.9 : {p999:>8.1f}")
print(f"Máximo absoluto: {max_val:>8.1f}")

# 5. Ver cuántos estudiantes se ven afectados por el clipping actual
n_clipped = (stats > p995).sum()
print(f"\nEstudiantes que serán 'capados' con 0.995: {n_clipped} ({n_clipped/len(stats)*100:.2f}%)")

# 6. Ver la diferencia real de los "locos"
top_1 = stats.max()
ratio = top_1 / p995 if p995 > 0 else 0
print(f"El estudiante #1 tiene {ratio:.1f} veces más clicks que el punto de corte.")
