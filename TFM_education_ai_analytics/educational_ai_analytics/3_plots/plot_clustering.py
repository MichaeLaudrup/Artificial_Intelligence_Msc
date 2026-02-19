import os
import warnings

# Silenciar warnings de Protobuf y logs de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import typer

import json
from educational_ai_analytics.config import (
    REPORTS_DIR,
    EMBEDDINGS_DATA_DIR,
    FEATURES_DATA_DIR,
    MODELS_DIR,
    CLUSTERING_MODELS_DIR,
    CLUSTERING_REPORTS_DIR,
    W_WINDOWS,
)
from .style import set_style

app = typer.Typer(help="Visualizaciones para Clustering.")
set_style()


def _apply_plot_theme(dark_mode: bool = True):
    import seaborn as sns
    if dark_mode:
        plt.style.use("dark_background")
        sns.set_theme(style="darkgrid", context="notebook")
        plt.rcParams.update({
            "figure.facecolor": "#101418",
            "axes.facecolor": "#151a21",
            "axes.edgecolor": "#D8DEE9",
            "axes.labelcolor": "#ECEFF4",
            "xtick.color": "#D8DEE9",
            "ytick.color": "#D8DEE9",
            "text.color": "#ECEFF4",
            "grid.color": "#3B4252",
            "legend.facecolor": "#1f2630",
            "legend.edgecolor": "#4C566A",
        })
    else:
        set_style()

@app.command("outcomes")
def clustering_outcomes_by_split(
    splits: str = typer.Option("training,validation,test", help="Splits a procesar separados por coma."),
    segmentation_name: str = typer.Option("segmentation_gmm_ae.csv", help="Nombre del archivo de segmentación."),
    window: int = typer.Option(max(W_WINDOWS) if W_WINDOWS else 24, help="Ventana temporal a usar."),
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio de salida para los reportes."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
    dark_mode: bool = typer.Option(True, "--dark-mode/--light-mode", help="Tema de gráficos."),
):
    """
    Generas gráficas de distribución de resultados académicos por clúster (OULAD).
    """
    _run_outcomes(splits, segmentation_name, window, out_dir, seed, dark_mode)

@app.command("profiles")
def clustering_profiles(
    segmentation_name: str = typer.Option("segmentation_gmm_ae.csv", help="Nombre del archivo de segmentación."),
    window: int = typer.Option(max(W_WINDOWS) if W_WINDOWS else 24, help="Ventana temporal a usar."),
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio de salida."),
    dark_mode: bool = typer.Option(True, "--dark-mode/--light-mode", help="Tema de gráficos."),
):
    """
    Genera un mapa de calor visualizando el perfil de cada clúster según sus variables.
    (Solo para el split de TRAINING).
    """
    _run_profiles(segmentation_name, window, out_dir, dark_mode)

@app.command("diagnose-modules")
def diagnose_modules(
    segmentation_name: str = typer.Option("segmentation_gmm_ae.csv", help="Nombre del archivo de segmentación."),
    window: int = typer.Option(max(W_WINDOWS) if W_WINDOWS else 24, help="Ventana temporal a usar."),
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio de salida."),
):
    """
    Analiza si los clústeres están dominados por módulos específicos (diagnóstico de sesgo).
    """
    _run_module_diagnosis(segmentation_name, window, out_dir)

@app.command("info")
def info():
    """Muestra información sobre este módulo de visualización."""
    logger.info("Módulo de visualización de clustering para TFM.")


@app.command("clean")
def clean_reports(
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio base de reportes de clustering."),
):
    """Limpia archivos residuales en la raíz de reports/clustering (conserva carpetas)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
            removed += 1
    logger.success(f"🧹 Limpieza completada en {out_dir} | archivos eliminados: {removed}")


@app.command("weekly-reports")
def weekly_reports(
    segmentation_name: str = typer.Option("segmentation_gmm_ae.csv", help="Nombre del archivo de segmentación."),
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio base de salida."),
    seed: int = typer.Option(42, help="Semilla aleatoria."),
    dark_mode: bool = typer.Option(True, "--dark-mode/--light-mode", help="Tema de gráficos."),
):
    """Genera informes por semana en reports/clustering/weekly/wXX y deja CSV intermedios en _intermediate."""
    weekly_root = out_dir / "weekly"
    interm_root = out_dir / "_intermediate"
    weekly_root.mkdir(parents=True, exist_ok=True)
    interm_root.mkdir(parents=True, exist_ok=True)

    for w in sorted([int(x) for x in W_WINDOWS]):
        w_dir = weekly_root / f"w{w:02d}"
        w_dir.mkdir(parents=True, exist_ok=True)
        w_intermediate = interm_root / f"w{w:02d}"
        w_intermediate.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Generando informe semanal W{w:02d}...")
        _run_outcomes("training,validation,test", segmentation_name, w, w_dir, seed, dark_mode)
        _run_profiles(segmentation_name, w, w_dir, dark_mode)
        _run_module_diagnosis(segmentation_name, w, w_intermediate)

    _run_module_success_verification(interm_root, dark_mode)
    logger.success(f"✅ Informes semanales listos en: {weekly_root}")

def _get_segmentation_path(split: str, name: str, window: int) -> Path:
    # 1. Mirar en la raíz de embeddings (estilo antiguo)
    p1 = EMBEDDINGS_DATA_DIR / split / name
    if p1.exists(): return p1
    # 2. Mirar en la subcarpeta de la ventana (estilo nuevo/multi-W)
    p2 = EMBEDDINGS_DATA_DIR / split / f"upto_w{int(window):02d}" / name
    return p2

def _get_dynamic_mapping(window: int | None = None):
    mapping_path = None
    if window is not None:
        p_window = CLUSTERING_MODELS_DIR / f"cluster_mapping_w{int(window):02d}.json"
        if p_window.exists():
            mapping_path = p_window
    if mapping_path is None:
        mapping_path = CLUSTERING_MODELS_DIR / "cluster_mapping.json"
    if not mapping_path.exists():
        mapping_path = MODELS_DIR / "cluster_mapping.json"
    mapping = {}
    if mapping_path.exists():
        try:
            with open(mapping_path, "r") as f:
                full_mp = json.load(f)
            mapping = {
                int(k): {
                    "name": v.get("name", f"Cluster {k}"),
                    "rank": v.get("rank_worst_to_best", int(k))
                } 
                for k, v in full_mp.items() if k.isdigit()
            }
        except Exception as e:
            logger.warning(f"Error cargando mapping: {e}")
    return mapping

def _run_profiles(segmentation_name, window, out_dir, dark_mode: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_theme(dark_mode)
    
    # Solo trabajamos con training para el perfilado base
    split = "training"
    seg_path = _get_segmentation_path(split, segmentation_name, window)
    day0_path = FEATURES_DATA_DIR / split / "day0_static_features.csv"
    dyn_path = FEATURES_DATA_DIR / split / "ae_uptow_features" / f"ae_uptow_features_w{int(window):02d}.csv"

    if not seg_path.exists() or not day0_path.exists() or not dyn_path.exists():
        logger.error(f"Faltan archivos para el perfilado en {split} (seg/day0/dyn).")
        return

    df_seg = pd.read_csv(seg_path, index_col=0)
    df_day0 = pd.read_csv(day0_path, index_col=0)
    df_dyn = pd.read_csv(dyn_path, index_col=0)
    df_feat = pd.concat([df_day0, df_dyn.reindex(df_day0.index)], axis=1).fillna(0.0)
    
    mapping = _get_dynamic_mapping(window)
    if mapping:
        if "cluster_name" not in df_seg.columns:
            df_seg["cluster_name"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("name", f"Cluster {int(cid)}"))
        df_seg["rank"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("rank", int(cid)))
    else:
        if "cluster_name" not in df_seg.columns:
            df_seg["cluster_name"] = [f"Cluster {int(c)}" for c in df_seg["cluster_id"]]
        df_seg["rank"] = df_seg["cluster_id"]

    # Unimos para tener features y clusters
    df = df_feat.join(df_seg[["cluster_name", "rank"]], how="inner")

    # Seleccionamos variables clave para el heatmap (agrupadas por temática)
    # Nota: Usamos las ya estandarizadas que vienen en engineered_features para que el heatmap sea comparable
    cols_to_plot = [
        # Dinámica Temporal y Fatiga
        "early_weeks_ratio", "effort_slope", "score_slope", "score_std",
        # Engagement General
        "total_weighted_engagement", "active_weeks", "activity_diversity",
        # Académico
        "avg_score", "pass_ratio", "submission_count", "late_ratio",
        # Contexto
        "studied_credits", "imd_band"
    ]
    # Aseguramos que existan
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]

    # Calculamos la media por cluster
    cluster_profiles = df.groupby(["cluster_name", "rank"])[cols_to_plot].mean().reset_index()
    cluster_profiles = cluster_profiles.sort_values("rank").set_index("cluster_name").drop(columns="rank")

    # Mapeo de nombres técnicos a "Human-Friendly"
    friendly_names = {
        "early_weeks_ratio": "Procrastinación (Ratio Inicial)",
        "effort_slope": "Pendiente Esfuerzo (Desaceleración)",
        "score_slope": "Tendencia Académica",
        "score_std": "Inconsistencia Notas (Std)",
        "total_weighted_engagement": "Volumen Interacción",
        "active_weeks": "Semanas Activo",
        "activity_diversity": "Diversidad Actividades",
        "avg_score": "Nota Media",
        "pass_ratio": "Ratio Aprobado",
        "submission_count": "Nº Entregas",
        "late_ratio": "Ratio Retrasos",
        "studied_credits": "Carga Créditos",
        "imd_band": "Vulnerabilidad (IMD)"
    }
    cluster_profiles = cluster_profiles.rename(columns=friendly_names)

    # Graficamos el Heatmap
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    
    # Usamos una paleta divergente (RdYlGn: Rojo-Amarillo-Verde) 
    # Como las variables ya están centradas en el preprocesamiento, 0 es la media global.
    sns.heatmap(
        cluster_profiles, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        center=0, 
        cbar_kws={'label': 'Desviación sobre la media (Z-score)'},
        linewidths=.5
    )
    
    plt.title("Perfil Característico de cada Clúster (Training Set)", fontsize=15, pad=20)
    plt.xlabel("Variables (Estandarizadas)", fontsize=11)
    plt.ylabel("Clústeres (Ordenados por rendimiento)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    
    out_file = out_dir / "clustering_feature_profiles.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    logger.success(f"📈 Perfiles de clúster guardados en: {out_file}")

    # -------------------------------------------------
    # Matriz Z-Effect (Top 30 variables) estilo notebook
    # -------------------------------------------------
    df_z = df_feat.join(df_seg[["cluster_name", "rank"]], how="inner")
    num_cols = df_z.select_dtypes(include=[np.number]).columns.tolist()
    for meta_col in ["final_result", "rank", "cluster_id"]:
        if meta_col in num_cols:
            num_cols.remove(meta_col)

    if len(num_cols) == 0:
        logger.warning("No hay columnas numéricas para calcular Z-Effect.")
        return

    base = df_z[num_cols].copy()
    global_mean = base.mean(axis=0)
    global_std = base.std(axis=0).replace(0, 1.0)

    by_cluster = (
        df_z.groupby(["cluster_name", "rank"])[num_cols]
        .mean()
        .reset_index()
        .sort_values("rank")
        .set_index("cluster_name")
        .drop(columns="rank")
    )

    effect = (by_cluster - global_mean) / global_std
    effect = effect.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    top_n = min(30, effect.shape[1])
    top_vars = (effect.max() - effect.min()).sort_values(ascending=False).head(top_n).index
    matrix = effect[top_vars].T

    plt.figure(figsize=(12, max(8, int(0.35 * len(top_vars)))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Z-Effect"},
        annot_kws={"size": 7},
        linewidths=0.3,
    )
    plt.title(f"Matriz Z-Effect (Top {top_n} variables) | W{int(window):02d}", fontsize=14)
    plt.xlabel("Clústeres")
    plt.ylabel("Variables")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(fontsize=8)

    out_z = out_dir / f"clustering_z_effect_top{top_n}_w{int(window):02d}.png"
    plt.savefig(out_z, dpi=220, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    logger.success(f"🧬 Matriz Z-Effect guardada en: {out_z}")

def _run_module_diagnosis(segmentation_name, window, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    split = "training"
    seg_path = _get_segmentation_path(split, segmentation_name, window)
    
    if not seg_path.exists():
        logger.error(f"No existe segmentation en {seg_path}")
        return

    df_seg = pd.read_csv(seg_path, index_col=0)
    mapping = _get_dynamic_mapping(window)
    
    # Extraer módulo y presentación del unique_id (formato: id_modulo_presentacion)
    # Ejemplo: 11391_AAA_2013J
    def parse_uid(uid):
        parts = str(uid).split("_")
        if len(parts) >= 3:
            return parts[1], parts[2], f"{parts[1]}_{parts[2]}"
        return "Unknown", "Unknown", "Unknown"

    parsed = [parse_uid(uid) for uid in df_seg.index]
    df_seg["code_module"] = [p[0] for p in parsed]
    df_seg["code_presentation"] = [p[1] for p in parsed]
    df_seg["module_presentation"] = [p[2] for p in parsed]
    
    if mapping:
        if "cluster_name" not in df_seg.columns:
            df_seg["cluster_name"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("name", f"Cluster {int(cid)}"))
        df_seg["rank"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("rank", int(cid)))
    else:
        if "cluster_name" not in df_seg.columns:
            df_seg["cluster_name"] = [f"Cluster {int(c)}" for c in df_seg["cluster_id"]]
        df_seg["rank"] = df_seg["cluster_id"]

    # Generar tabla de diagnóstico
    diagnosis_data = []
    clusters = sorted(df_seg["cluster_name"].unique(), key=lambda x: df_seg[df_seg["cluster_name"]==x]["rank"].iloc[0])

    logger.info("🩺 Iniciando diagnóstico de sesgo por módulo...")
    
    for cname in clusters:
        sub = df_seg[df_seg["cluster_name"] == cname]
        n_cluster = len(sub)
        
        # Top 10 módulos en este clúster
        top_mods = sub["module_presentation"].value_counts().head(10)
        
        for mod, count in top_mods.items():
            pct = (count / n_cluster) * 100.0
            diagnosis_data.append({
                "cluster": cname,
                "module_presentation": mod,
                "count": count,
                "pct_within_cluster": pct
            })

    df_diag = pd.DataFrame(diagnosis_data)
    out_file = out_dir / "clustering_module_bias_diagnosis.csv"
    df_diag.to_csv(out_file, index=False)
    
    logger.success(f"🩺 Diagnóstico guardado en: {out_file}")
    
    # Mostrar resumen por consola para el usuario
    for cname in clusters:
        print(f"\n--- Clúster: {cname} ---")
        c_diag = df_diag[df_diag["cluster"] == cname].head(3)
        for _, row in c_diag.iterrows():
            print(f"  > {row['module_presentation']}: {row['pct_within_cluster']:.1f}% ({row['count']} est.)")

@app.command("verify-module-success")
def verify_module_success(
    out_dir: Path = typer.Option(CLUSTERING_REPORTS_DIR, help="Directorio de salida."),
    dark_mode: bool = typer.Option(True, "--dark-mode/--light-mode", help="Tema de gráficos."),
):
    """
    Calcula la tasa de éxito (Distinction/Pass) por módulo globalmente para detectar sesgos.
    """
    _run_module_success_verification(out_dir, dark_mode)

def _run_module_success_verification(out_dir, dark_mode: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_theme(dark_mode)
    split = "training"
    tgt_path = FEATURES_DATA_DIR / split / "target.csv"
    
    if not tgt_path.exists():
        logger.error(f"No existe target.csv en {tgt_path}")
        return

    df_tgt = pd.read_csv(tgt_path, index_col=0)
    
    # Extraer módulo
    df_tgt["module"] = [str(uid).split("_")[1] for uid in df_tgt.index]
    df_tgt["presentation"] = [str(uid).split("_")[2] for uid in df_tgt.index]
    df_tgt["mod_pres"] = df_tgt["module"] + "_" + df_tgt["presentation"]

    # Calcular tasas
    # 3: Distinction, 2: Pass
    stats = df_tgt.groupby("mod_pres")["final_result"].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    # Renombrar columnas si existen
    rename_cols = {3: "Distinction (%)", 2: "Pass (%)", 1: "Fail (%)", 0: "Withdrawn (%)"}
    stats = stats.rename(columns=rename_cols)
    
    # Asegurar que todas las columnas existan
    for col in rename_cols.values():
        if col not in stats.columns:
            stats[col] = 0.0

    stats = stats.sort_values("Distinction (%)", ascending=False)
    
    # Guardar CSV
    out_csv = out_dir / "module_success_rates.csv"
    stats.to_csv(out_csv)
    logger.success(f"📊 Tasas por módulo guardadas en: {out_csv}")

    # Graficar
    plt.figure(figsize=(14, 7))
    colors = ["#3DDC97", "#64B5F6", "#FFB74D", "#EF5350"]
    
    # Usar solo las columnas de éxito para el plot
    plot_cols = ["Distinction (%)", "Pass (%)", "Fail (%)", "Withdrawn (%)"]
    stats[plot_cols].plot(kind="bar", stacked=True, color=colors, ax=plt.gca(), width=0.8)
    
    plt.title("Tasa de Resultados por Módulo/Presentación (Training Global)", fontsize=15)
    plt.ylabel("Porcentaje (%)")
    plt.xlabel("Módulo_Presentación")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Leyenda')
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    out_png = out_dir / "module_success_rates.png"
    plt.savefig(out_png, dpi=200, facecolor=plt.gcf().get_facecolor())
    plt.close()
    logger.success(f"🖼️ Gráfico de éxito por módulo: {out_png}")

def _run_outcomes(splits, segmentation_name, window, out_dir, seed, dark_mode: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_theme(dark_mode)

    mapping = _get_dynamic_mapping(window)
    if mapping:
        logger.info(f"Cargado mapping dinámico para visualización.")

    # mapping: 0 Withdrawn, 1 Fail, 2 Pass, 3 Distinction
    id2name = {0: "Withdrawn", 1: "Fail", 2: "Pass", 3: "Distinction"}
    categories = ["Distinction", "Pass", "Fail", "Withdrawn"]

    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    
    def _load_split(split: str):
        seg_path = _get_segmentation_path(split, segmentation_name, window)
        tgt_path = FEATURES_DATA_DIR / split / "target.csv"

        if not seg_path.exists():
            logger.error(f"[{split}] No existe segmentation: {seg_path}")
            return None
        if not tgt_path.exists():
            logger.error(f"[{split}] No existe target.csv: {tgt_path}")
            return None

        df_seg = pd.read_csv(seg_path, index_col=0).sort_index()
        df_tgt = pd.read_csv(tgt_path, index_col=0).sort_index()

        common = df_seg.index.intersection(df_tgt.index)
        df_seg = df_seg.loc[common].copy()
        df_tgt = df_tgt.loc[common].copy()

        # Sobrescribo o asigno nombres usando el mapping dinámico del JSON
        if mapping:
            if "cluster_name" not in df_seg.columns:
                df_seg["cluster_name"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("name", f"Cluster {int(cid)}"))
            df_seg["cluster_rank"] = df_seg["cluster_id"].map(lambda cid: mapping.get(int(cid), {}).get("rank", 99))
        else:
            df_seg["cluster_name"] = df_seg.get("cluster_name", df_seg.get("cluster_label", df_seg.get("cluster_id"))).astype(str)
            df_seg["cluster_rank"] = df_seg["cluster_id"]

        df_tgt["actual_result_name"] = df_tgt["final_result"].map(id2name).fillna("Withdrawn")
        df_tgt["is_success"] = df_tgt["final_result"].isin([2, 3]).astype(int)

        df = df_seg.join(df_tgt[["final_result", "actual_result_name", "is_success"]], how="inner")
        return df

    def _compute_table(df: pd.DataFrame):
        # % por cluster (usando el nombre dinámico del JSON) y categoría
        tab = pd.crosstab(df["cluster_name"], df["actual_result_name"], normalize="index") * 100.0
        for c in categories:
            if c not in tab.columns:
                tab[c] = 0.0
        tab = tab[categories]
        
        # Necesito el n y el rank para ordenar
        stats = df.groupby("cluster_name").agg(
            n=("cluster_id", "size"),
            rank=("cluster_rank", "first")
        )
        tab = tab.join(stats)
        
        # Ordeno por el ranking pedagógico definido en training_clustering
        tab = tab.sort_values("rank", ascending=True)
        
        global_success = float(df["is_success"].mean() * 100.0)
        return tab, global_success

    def _plot_one(ax, tab, global_success, title_str: str):
        x = np.arange(len(tab))
        width = 0.20
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        bar_cfg = [
            ("Distinction", "Excelencia (Distinction)", "#3DDC97"),
            ("Pass", "Aprobado (Pass)", "#64B5F6"),
            ("Fail", "Suspenso (Fail)", "#FFB74D"),
            ("Withdrawn", "Abandono (Withdrawn)", "#EF5350"),
        ]
        for i, (col, label, color) in enumerate(bar_cfg):
            ax.bar(x + offsets[i], tab[col].values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5)
        
        ax.axhline(global_success, color="#B0BEC5", linestyle="--", linewidth=1.5, label=f"Media Global Éxito ({global_success:.1f}%)")
        ax.set_title(title_str, fontsize=13)
        ax.set_ylabel("Porcentaje (%)")
        ax.set_ylim(0, 110)
        ax.set_xticks(x)
        ax.set_xticklabels(tab.index.values, rotation=25, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        for i, n_i in enumerate(tab["n"].values):
            y_max = float(max(tab.iloc[i][c] for c in categories))
            ax.text(i, y_max + 3, f"n={int(n_i)}", ha="center", va="bottom", fontsize=9, fontweight="bold", alpha=0.7)

    per_split_tabs = []
    for split in split_list:
        df = _load_split(split)
        if df is None: continue
        tab, global_success = _compute_table(df)
        per_split_tabs.append((split, tab, global_success))

        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_one(ax, tab, global_success, title_str=f"Distribución de Resultados por Clúster — {split.upper()}")
        ax.legend(loc="upper right", frameon=True, fontsize=9, title="Leyenda (color → resultado)")
        out_file = out_dir / f"clustering_outcomes_{split}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
        plt.close()
        logger.success(f"📊 Guardado: {out_file}")

    if len(per_split_tabs) > 0:
        nrows = len(per_split_tabs)
        fig, axes = plt.subplots(nrows, 1, figsize=(13, 5 * nrows), constrained_layout=True)
        if nrows == 1: axes = [axes]
        for ax, (split, tab, global_success) in zip(axes, per_split_tabs):
            _plot_one(ax, tab, global_success, title_str=f"{split.upper()}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=True, fontsize=9, title="Leyenda")
        out_file_all = out_dir / "clustering_outcomes_all_splits.png"
        plt.savefig(out_file_all, dpi=200, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
        plt.close()
        logger.success(f"📌 Guardado combinado: {out_file_all}")

if __name__ == "__main__":
    app()
