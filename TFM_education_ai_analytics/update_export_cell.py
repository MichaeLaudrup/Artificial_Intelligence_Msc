import json

notebook_path = '/workspace/TFM_education_ai_analytics/notebooks/3__clustering_exploration.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell starts with # --- CELDA FINAL: Exportación de Resultados para el TFM ---
target_cell_start = "# --- CELDA FINAL: Exportación de Resultados para el TFM ---"

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any(target_cell_start in line for line in cell['source']):
        cell['source'] = [
            "# --- CELDA FINAL: Exportación de Resultados para el TFM ---\n",
            "\n",
            "# 1. Preparar el DataFrame de salida\n",
            "df_final_results = df_static.copy()\n",
            "\n",
            "# 2. Asignar los clusters ganadores\n",
            "df_final_results['cluster_id'] = clusters_gmm\n",
            "df_final_results['probabilidad_perfil_asignado'] = max_probs\n",
            "df_final_results['resultado_real'] = df_target['final_result']\n",
            "\n",
            "# 3. Mapear nombres a los clusters\n",
            "nombres_perfiles = {\n",
            "    0: \"En Riesgo Extremo\",\n",
            "    1: \"Excelencia / Elite\",\n",
            "    2: \"Promedio (Baja Actividad)\",\n",
            "    3: \"Promedio (Constante)\",\n",
            "    4: \"Procrastinador Potencial\",\n",
            "    5: \"Alta Actividad / Disperso\"\n",
            "}\n",
            "df_final_results['perfil_pedagogico'] = df_final_results['cluster_id'].map(nombres_perfiles)\n",
            "\n",
            "# 4. Añadir Probabilidades detalladas de cada perfil (Soft Clustering)\n",
            "# Usamos 'gmm_tuned' y 'X_ae_scaled' que vienen de la celda de mejora\n",
            "probs_detalladas = gmm_tuned.predict_proba(X_ae_scaled)\n",
            "\n",
            "for i in range(K_FINAL):\n",
            "    nombre_col = f\"Prob_{nombres_perfiles[i]}\"\n",
            "    df_final_results[nombre_col] = probs_detalladas[:, i]\n",
            "\n",
            "# 5. Guardar en CSV\n",
            "OUTPUT_PATH = BASE_DIR / \"data/5_results/student_clusters_final.csv\"\n",
            "OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)\n",
            "df_final_results.to_csv(OUTPUT_PATH)\n",
            "\n",
            "print(f\"🎉 ¡TRABAJO COMPLETADO!\")\n",
            "print(f\"✅ Se han añadido {K_FINAL} columnas de probabilidad para estudio de riesgo fronterizo.\")\n",
            "print(f\"Resultados guardados en: {OUTPUT_PATH}\")\n",
            "print(f\"Total alumnos procesados: {len(df_final_results)}\")\n",
            "print(\"\\nDistribución final de perfiles:\")\n",
            "print(df_final_results['perfil_pedagogico'].value_counts())"
        ]
        # Clear output to allow fresh run
        cell['outputs'] = []
        cell['execution_count'] = None
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
