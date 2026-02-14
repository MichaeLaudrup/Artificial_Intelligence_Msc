import json

file_path = '/workspace/TFM_education_ai_analytics/notebooks/3__clustering_exploration.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6) Entrenamiento de Clústeres Finales (K_PCA=4, K_AE=5)\n",
            "\n",
            "Una vez analizadas las curvas de inercia y silhouette, procedemos a realizar el clustering final con los valores de k seleccionados para cada espacio de representación."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.cluster import KMeans\n",
            "from sklearn.metrics import silhouette_score, calinski_harabasz_score\n",
            "\n",
            "# 1) Parámetros seleccionados\n",
            "K_PCA = 4\n",
            "K_AE = 5\n",
            "SEED = 42\n",
            "\n",
            "# 2) Ajuste de KMeans\n",
            "print(f\"⏳ Entrenando KMeans: PCA (k={K_PCA}) y Autoencoder (k={K_AE})...\")\n",
            "\n",
            "km_pca = KMeans(n_clusters=K_PCA, init=\"k-means++\", n_init=10, random_state=SEED)\n",
            "labels_pca = km_pca.fit_predict(X_pca)\n",
            "\n",
            "km_ae = KMeans(n_clusters=K_AE, init=\"k-means++\", n_init=10, random_state=SEED)\n",
            "labels_ae = km_ae.fit_predict(X_ae)\n",
            "\n",
            "# 3) Cálculo de métricas\n",
            "def print_metrics(X, labels, name):\n",
            "    # Silhouette con sample si es grande\n",
            "    sil = silhouette_score(X, labels, sample_size=10000, random_state=SEED)\n",
            "    cal = calinski_harabasz_score(X, labels)\n",
            "    print(f\"\\n📊 Métricas para {name}:\")\n",
            "    print(f\"  - Silhouette Score: {sil:.4f}\")\n",
            "    print(f\"  - Calinski-Harabasz: {cal:.2f}\")\n",
            "\n",
            "print_metrics(X_pca, labels_pca, f\"PCA (k={K_PCA})\")\n",
            "print_metrics(X_ae, labels_ae, f\"Autoencoder (k={K_AE})\")\n",
            "\n",
            "# 4) Visualización (Reutilizando el estilo previo)\n",
            "# Extraemos etiquetas para el mismo sample usado en la visualización anterior\n",
            "labels_ae_s = labels_ae[sample_idx]\n",
            "labels_pca_s = labels_pca[sample_idx]\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)\n",
            "fig.suptitle(\n",
            "    f\"Resultados de Clustering: Espacio Latente AE (k={K_AE}) vs PCA (k={K_PCA})\\nN={len(sample_idx)} estudiantes\",\n",
            "    fontsize=16\n",
            ")\n",
            "\n",
            "# A) Autoencoder\n",
            "scatter_ae = axes[0].scatter(p2_ae[:, 0], p2_ae[:, 1], c=labels_ae_s, s=6, alpha=0.5, cmap=\"Spectral\")\n",
            "axes[0].set_title(f\"Autoencoder (k={K_AE}) - Clústeres en 2D (PCA)\\nVar. Explicada: {var_ae:.2%}\", fontsize=14)\n",
            "plt.colorbar(scatter_ae, ax=axes[0], label=\"Cluster ID\", ticks=range(K_AE))\n",
            "\n",
            "# B) PCA\n",
            "scatter_pca = axes[1].scatter(p2_pca[:, 0], p2_pca[:, 1], c=labels_pca_s, s=6, alpha=0.5, cmap=\"Spectral\")\n",
            "axes[1].set_title(f\"PCA (k={K_PCA}) - Clústeres en 2D (PCA)\\nVar. Explicada: {var_pca:.2%}\", fontsize=14)\n",
            "plt.colorbar(scatter_pca, ax=axes[1], label=\"Cluster ID\", ticks=range(K_PCA))\n",
            "\n",
            "for ax in axes:\n",
            "    ax.set_xlabel(\"PC1\")\n",
            "    ax.set_ylabel(\"PC2\")\n",
            "    ax.set_xlim(xlim)\n",
            "    ax.set_ylim(ylim)\n",
            "    ax.grid(True, alpha=0.2)\n",
            "\n",
            "plt.show()"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
