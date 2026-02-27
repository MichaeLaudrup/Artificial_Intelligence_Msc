"""
training_reporter.py
====================
Módulo de reporte visual del entrenamiento del Autoencoder.

Responsabilidades:
  · TrainingMetricsCollector — acumula métricas época a época (fase 3).
  · plot_training_evolution  — renderiza y guarda la figura compuesta en reports/ae/.

Uso desde train_autoencoder.py:
    from .training_reporter import TrainingMetricsCollector, plot_training_evolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import numpy as np
from loguru import logger

# ─── Paleta ─────────────────────────────────────────────────────────────────────
DARK_BG       = "#0F1117"
PANEL_BG      = "#1A1D27"
ACCENT_RECON  = "#7C83FD"   # azul-violeta  — reconstrucción
ACCENT_KL     = "#FC5C7D"   # rosa-coral    — clustering / KL
ACCENT_VAL    = "#43E97B"   # verde-menta   — validación
ACCENT_OBJ    = "#F8D800"   # amarillo      — objetivo
ACCENT_SMOOTH = "#FFFFFF"   # blanco        — curva suavizada
GRID_COLOR    = "#2A2D3A"
TEXT_COLOR    = "#E8EAED"
MUTED_ALPHA   = 0.20        # opacidad de la curva ruidosa de fondo


# ─── Colector de métricas ────────────────────────────────────────────────────────

@dataclass
class TrainingMetricsCollector:
    """
    Acumula las métricas de la fase 3 (joint training) época a época.
    Llamar a ``record(...)`` al final de cada época.
    """
    epochs:          list[int]   = field(default_factory=list)
    train_recon:     list[float] = field(default_factory=list)
    train_kl_raw:    list[float] = field(default_factory=list)
    train_kl_scaled: list[float] = field(default_factory=list)
    train_obj:       list[float] = field(default_factory=list)
    val_recon:       list[float] = field(default_factory=list)
    val_kl_raw:      list[float] = field(default_factory=list)
    model_obj:       list[float] = field(default_factory=list)

    def record(
        self,
        *,
        epoch: int,
        train_recon: float,
        train_kl_raw: float,
        train_kl_scaled: float,
        train_obj: float,
        val_recon: float,
        val_kl_raw: float,
        model_obj: float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_recon.append(train_recon)
        self.train_kl_raw.append(train_kl_raw)
        self.train_kl_scaled.append(train_kl_scaled)
        self.train_obj.append(train_obj)
        self.val_recon.append(val_recon)
        self.val_kl_raw.append(val_kl_raw)
        self.model_obj.append(model_obj)

    @property
    def best_epoch(self) -> Optional[int]:
        """Época (1-indexed) con el menor ``model_obj``."""
        return self.epochs[int(np.argmin(self.model_obj))] if self.model_obj else None

    @property
    def last_epoch(self) -> Optional[int]:
        return self.epochs[-1] if self.epochs else None


# ─── Utilidades de suavizado ─────────────────────────────────────────────────────

def _ema(values: list[float], alpha: float = 0.3) -> np.ndarray:
    """Exponential Moving Average de una serie."""
    arr = np.array(values, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_min(values: list[float], window: int = 5) -> np.ndarray:
    """Mínimo acumulado (running minimum) — muestra la tendencia real del mejor objetivo."""
    arr = np.array(values, dtype=float)
    return np.minimum.accumulate(arr)


def _plot_with_ema(
    ax: plt.Axes,
    x: list,
    y: list[float],
    color: str,
    label: str,
    alpha: float = 0.3,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    """Dibuja la curva cruda semitransparente + su EMA suavizada encima."""
    ax.plot(x, y, color=color, linewidth=0.8, alpha=MUTED_ALPHA,
            linestyle=linestyle, marker=None)
    smooth = _ema(y, alpha=alpha)
    ax.plot(x, smooth, color=color, linewidth=2.2,
            linestyle=linestyle, label=label,
            marker=marker, markersize=3, markevery=max(1, len(x) // 12))


# ─── Helpers de estilo ───────────────────────────────────────────────────────────

def _style_ax(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "Época",
    ylabel: str = "Pérdida",
) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8.5)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8.5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7.5)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)


def _legend(ax: plt.Axes) -> None:
    ax.legend(
        fontsize=7.5, framealpha=0.3,
        facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
    )


def _blank_panel(ax: plt.Axes, message: str, title: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=TEXT_COLOR, fontsize=11, transform=ax.transAxes)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


def _annotate_epoch(ax: plt.Axes, epoch: int, label: str, color: str, yrel: float = 0.92) -> None:
    """Línea vertical + etiqueta de texto para un epoch especial."""
    ax.axvline(epoch, color=color, linestyle="--", linewidth=1.1, alpha=0.75)
    ylim = ax.get_ylim()
    y_pos = ylim[0] + yrel * (ylim[1] - ylim[0])
    ax.text(epoch + 0.3, y_pos, label, color=color,
            fontsize=7, va="top", ha="left", alpha=0.9)


# ─── Paneles individuales ────────────────────────────────────────────────────────

def _panel_pretrain(ax: plt.Axes, pretrain_history: dict) -> None:
    """Panel 1 — Fase 1: pérdida de reconstrucción train/val con EMA."""
    epochs = list(range(1, len(pretrain_history.get("loss", [])) + 1))
    if not epochs:
        _blank_panel(ax, "Sin datos de pretrain", "Fase 1 — Pre-entrenamiento")
        return

    _plot_with_ema(ax, epochs, pretrain_history["loss"],
                   ACCENT_RECON, "Train (Huber)", marker="o")

    val_loss = pretrain_history.get("val_loss", [])
    if val_loss:
        _plot_with_ema(ax, epochs, val_loss,
                       ACCENT_VAL, "Val (Huber)", marker="s", linestyle="--")

    _style_ax(ax, "Fase 1 — Pre-entrenamiento\n(Pérdida de Reconstrucción)")
    _legend(ax)


def _panel_joint_losses(ax: plt.Axes, collector: TrainingMetricsCollector) -> None:
    """
    Panel 2 — Fase 3: reconstrucción (eje izq.) vs objetivo monitoreado (eje der.)
    usando twinx para separar las escalas y evitar solapamientos.
    """
    j = collector
    eps = j.epochs

    # ── Eje izquierdo: reconstrucción ──────────────────────────────────────
    _plot_with_ema(ax, eps, j.train_recon, ACCENT_RECON,
                   "Train Reconstruc.", marker="o")
    if any(v > 0 for v in j.val_recon):
        _plot_with_ema(ax, eps, j.val_recon, ACCENT_VAL,
                       "Val Reconstruc.", marker="s", linestyle="--")
    _style_ax(ax,
              "Fase 3 — Entrenamiento Conjunto\n(Reconstrucción vs Objetivo)",
              ylabel="Pérdida reconstrucción")

    # ── Eje derecho: objetivo monitoreado ──────────────────────────────────
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    _plot_with_ema(ax2, eps, j.model_obj, ACCENT_OBJ,
                   "Objetivo Monit. (ValObj)", marker="D", linestyle="-.")
    ax2.set_ylabel("Objetivo monitoreado", color=ACCENT_OBJ, fontsize=8.5)
    ax2.tick_params(colors=ACCENT_OBJ, labelsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID_COLOR)
    ax2.spines["left"].set_color(GRID_COLOR)
    ax2.spines["bottom"].set_color(GRID_COLOR)

    # ── Anotaciones ────────────────────────────────────────────────────────
    if j.best_epoch is not None:
        _annotate_epoch(ax, j.best_epoch, f"Mejor ({j.best_epoch})", ACCENT_OBJ)

    # ── Leyenda combinada ──────────────────────────────────────────────────
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)


def _panel_kl(
    ax: plt.Axes,
    collector: TrainingMetricsCollector,
    use_clustering_objective: bool,
) -> None:
    """
    Panel 3 — Fase 3: divergencia KL con EMA + nota si val KL sube.
    Si la escala de val_kl_raw es muy distinta a train_kl_raw, se usa
    un eje secundario (twinx) para evitar que val aparezca "plano en 0".
    """
    if not use_clustering_objective:
        _blank_panel(ax, "Clustering objetivo\nDESACTIVADO",
                     "Fase 3 — Pérdida de Clustering (KL)")
        return

    j = collector
    _plot_with_ema(ax, j.epochs, j.train_kl_raw,
                   ACCENT_KL, "Train KL raw", marker="o")
    _plot_with_ema(ax, j.epochs, j.train_kl_scaled,
                   ACCENT_OBJ, "Train KL escalado", marker="^", linestyle="-.")

    has_val_kl = any(v > 1e-9 for v in j.val_kl_raw)
    if has_val_kl:
        val_max  = max(j.val_kl_raw)
        train_max = max(j.train_kl_raw) if j.train_kl_raw else 1.0
        scale_ratio = train_max / max(val_max, 1e-12)

        if scale_ratio > 5.0:
            # ── Escala muy diferente → eje derecho para val ───────────────
            ax2_kl = ax.twinx()
            ax2_kl.set_facecolor("none")
            _plot_with_ema(ax2_kl, j.epochs, j.val_kl_raw,
                           ACCENT_VAL, "Val KL raw (eje ▶)", marker="s", linestyle="--")
            ax2_kl.set_ylabel("Val KL raw", color=ACCENT_VAL, fontsize=8)
            ax2_kl.tick_params(colors=ACCENT_VAL, labelsize=7)
            ax2_kl.spines["top"].set_visible(False)
            ax2_kl.spines["right"].set_color(GRID_COLOR)
            ax2_kl.spines["left"].set_color(GRID_COLOR)
            ax2_kl.spines["bottom"].set_color(GRID_COLOR)
            # Leyenda combinada
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2_kl.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=7, framealpha=0.3,
                      facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax.text(0.03, 0.95,
                    f"⚠ Escalas distintas (×{scale_ratio:.0f})\nVal KL en eje derecho",
                    transform=ax.transAxes, fontsize=6.5,
                    color=ACCENT_VAL, alpha=0.85, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                              edgecolor=GRID_COLOR, alpha=0.6))
        else:
            _plot_with_ema(ax, j.epochs, j.val_kl_raw,
                           ACCENT_VAL, "Val KL raw", marker="s", linestyle="--")
            _legend(ax)

        # Aviso tendencia alcista (esperado en DEC)
        trend = np.polyfit(j.epochs, j.val_kl_raw, 1)[0]
        if trend > 0:
            ax.text(0.97, 0.08,
                    "↑ Val KL creciente\n(esperado en DEC)",
                    transform=ax.transAxes, fontsize=6.5,
                    color=ACCENT_VAL, alpha=0.8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                              edgecolor=GRID_COLOR, alpha=0.6))
    else:
        # val_kl_raw está efectivamente en cero → diagnóstico
        ax.text(0.5, 0.5,
                "Val KL ≈ 0\n(blend bajo o P≈Q en val)",
                transform=ax.transAxes, fontsize=8,
                color=ACCENT_VAL, alpha=0.75, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                          edgecolor=ACCENT_VAL, alpha=0.5))
        _legend(ax)

    _style_ax(ax, "Fase 3 — Pérdida de Clustering\n(Divergencia KL)")


def _panel_convergence(ax: plt.Axes, collector: TrainingMetricsCollector) -> None:
    """
    Panel 4 — Convergencia:
      · Curva model_obj cruda (semitransparente).
      · EMA de model_obj (suavizada).
      · Running minimum (mejor objetivo alcanzado hasta cada época).
    """
    if len(collector.model_obj) < 2:
        _blank_panel(ax, "Sin datos de\nentrenamiento conjunto",
                     "Convergencia del Objetivo Monitoreado")
        return

    j = collector
    obj_arr = np.array(j.model_obj)

    # Cruda semitransparente
    ax.plot(j.epochs, obj_arr, color=ACCENT_OBJ, linewidth=0.8,
            alpha=MUTED_ALPHA, label=None)

    # EMA suavizada
    smooth = _ema(j.model_obj, alpha=0.3)
    ax.plot(j.epochs, smooth, color=ACCENT_OBJ, linewidth=2,
            linestyle="-", label="ValObj (EMA)",
            marker="o", markersize=3, markevery=max(1, len(j.epochs) // 12))

    # Running minimum — tendencia real de la mejor solución encontrada
    run_min = _rolling_min(j.model_obj)
    ax.fill_between(j.epochs, run_min, alpha=0.18, color=ACCENT_VAL)
    ax.plot(j.epochs, run_min, color=ACCENT_VAL, linewidth=1.8,
            linestyle="--", label="Mínimo acumulado")

    # Anotación early stopping
    if j.last_epoch is not None:
        ax.axvline(j.last_epoch, color=ACCENT_KL, linestyle=":",
                   linewidth=1.1, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(j.last_epoch - 0.3,
                ylim[0] + 0.88 * (ylim[1] - ylim[0]),
                f"Early Stop\n(ep. {j.last_epoch})",
                color=ACCENT_KL, fontsize=6.5, ha="right", va="top")

    # Anotación mejor época
    if j.best_epoch is not None:
        _annotate_epoch(ax, j.best_epoch,
                        f"Mejor ({j.best_epoch})", ACCENT_OBJ, yrel=0.78)

    _style_ax(ax, "Convergencia del Objetivo Monitoreado\n(ValObj + Mínimo Acumulado)",
              ylabel="Objetivo (ValObj)")
    _legend(ax)


# ─── API pública ─────────────────────────────────────────────────────────────────

def plot_training_evolution(
    pretrain_history: dict,
    collector: TrainingMetricsCollector,
    use_clustering_objective: bool,
    save_path: Path,
) -> None:
    """
    Genera y guarda ``save_path`` con 4 paneles ilustrativos:
      1. Fase 1 — Pérdida de reconstrucción pretrain (train/val, con EMA).
      2. Fase 3 — Reconstrucción (eje izq.) vs Objetivo monitoreado (eje der., twinx).
      3. Fase 3 — Divergencia KL (con EMA + aviso si val KL sube).
      4. Fase 3 — ValObj crudo + EMA + mínimo acumulado (convergencia real).

    Parameters
    ----------
    pretrain_history:
        ``history.history`` devuelto por ``model.fit(...)`` en la fase de pretrain.
    collector:
        Instancia de ``TrainingMetricsCollector`` rellena durante la fase joint.
    use_clustering_objective:
        Indica si el objetivo de clustering estuvo activo (controla el panel KL).
    save_path:
        Ruta completa de salida (p. ej. ``AE_REPORTS_DIR / "training_evolution.png"``).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(
        "Evolución del Entrenamiento — Autoencoder (DCN)",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
        y=0.98,   # alto para no pisar títulos de paneles
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.52, wspace=0.38,
        top=0.91, bottom=0.07, left=0.07, right=0.96,
    )

    _panel_pretrain(fig.add_subplot(gs[0, 0]), pretrain_history)

    has_joint = len(collector.epochs) > 0
    if has_joint:
        _panel_joint_losses(fig.add_subplot(gs[0, 1]), collector)
        _panel_kl(fig.add_subplot(gs[1, 0]), collector, use_clustering_objective)
        _panel_convergence(fig.add_subplot(gs[1, 1]), collector)
    else:
        for pos in [(0, 1), (1, 0), (1, 1)]:
            _blank_panel(fig.add_subplot(gs[pos]), "Sin datos de\nfase conjunta", "")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.success(f"📊 Gráficas guardadas → {save_path}")


# ─── UMAP / PCA de embeddings ─────────────────────────────────────────────────────────

def plot_embeddings_umap(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path,
    n_clusters: int,
    title_suffix: str = "",
) -> None:
    """
    Genera una figura 2-panel con la proyección 2D de los embeddings:
      - Panel izq.: UMAP (o PCA si umap no está instalado).
      - Panel der.: distribución de tamaños de cluster (barras).

    Parameters
    ----------
    embeddings:
        Array (N, latent_dim) de embeddings reales del encoder.
    cluster_labels:
        Array (N,) con el índice de cluster por muestra (argmax de Q).
    save_path:
        Ruta de salida para la imagen PNG.
    n_clusters:
        Número de clusters K.
    title_suffix:
        Texto adicional en el título (p.ej. "Best" o "Last").
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Reducción de dimensionalidad ────────────────────────────────
    method_label = "UMAP"
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
        logger.info("   🌏 UMAP proyección completada")
    except ImportError:
        # Fallback: PCA rápida
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        method_label = "PCA"
        logger.warning("   ⚠️ umap-learn no disponible → usando PCA como fallback. "
                       "Instala con: pip install umap-learn")

    # ── Paleta de colores ───────────────────────────────────────────
    palette = [
        "#7C83FD", "#43E97B", "#FC5C7D", "#F8D800",
        "#A8EDEA", "#FF9A9E", "#a29bfe", "#fd79a8",
        "#00cec9", "#e17055",
    ]
    cluster_colors = [palette[i % len(palette)] for i in range(n_clusters)]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG,
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        f"Espacio Latente del Autoencoder — {method_label} de Embeddings"
        + (f" ({title_suffix})" if title_suffix else ""),
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel izquierdo: scatter UMAP/PCA ─────────────────────────────
    ax_scatter = axes[0]
    ax_scatter.set_facecolor(PANEL_BG)

    for k in range(n_clusters):
        mask = cluster_labels == k
        if mask.sum() == 0:
            continue
        ax_scatter.scatter(
            coords[mask, 0], coords[mask, 1],
            c=cluster_colors[k], label=f"Cluster {k} (n={mask.sum()})",
            s=8, alpha=0.65, edgecolors="none",
        )
        # Centroide del cluster en el espacio UMAP
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax_scatter.text(
            cx, cy, str(k),
            fontsize=11, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25",
                      facecolor=cluster_colors[k], edgecolor="none", alpha=0.85),
        )

    ax_scatter.set_title(
        f"Proyección {method_label} — {n_clusters} clusters",
        color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8,
    )
    ax_scatter.set_xlabel(f"{method_label}-1", color=TEXT_COLOR, fontsize=9)
    ax_scatter.set_ylabel(f"{method_label}-2", color=TEXT_COLOR, fontsize=9)
    ax_scatter.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_scatter.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.4)
    for spine in ax_scatter.spines.values():
        spine.set_color(GRID_COLOR)
    ax_scatter.legend(
        fontsize=7.5, framealpha=0.3, markerscale=2.5,
        facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        bbox_to_anchor=(0.0, -0.13), loc="upper left", ncol=min(n_clusters, 5),
    )

    # ── Panel derecho: distribución de tamaños de cluster ─────────────
    ax_bar = axes[1]
    ax_bar.set_facecolor(PANEL_BG)

    counts = np.bincount(cluster_labels, minlength=n_clusters)
    pcts   = 100.0 * counts / max(counts.sum(), 1)
    ks     = list(range(n_clusters))
    bars   = ax_bar.barh(
        ks, pcts,
        color=[cluster_colors[k] for k in ks],
        edgecolor="none", height=0.65, alpha=0.85,
    )
    # Etiqueta con n y %
    for bar, k, pct, cnt in zip(bars, ks, pcts, counts):
        ax_bar.text(
            pct + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={cnt} ({pct:.1f}%)",
            va="center", ha="left", fontsize=8, color=TEXT_COLOR, alpha=0.9,
        )

    ax_bar.set_yticks(ks)
    ax_bar.set_yticklabels([f"Cluster {k}" for k in ks], color=TEXT_COLOR, fontsize=8.5)
    ax_bar.set_xlabel("% de muestras", color=TEXT_COLOR, fontsize=9)
    ax_bar.set_title(
        "Distribución de clusters",
        color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8,
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(pcts) * 1.35)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_bar.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--",
                alpha=0.4, axis="x")
    for spine in ax_bar.spines.values():
        spine.set_color(GRID_COLOR)

    # ── Aviso si hay colapso de cluster ─────────────────────────────
    max_frac = counts.max() / max(counts.sum(), 1)
    if max_frac > 0.80:
        fig.text(
            0.5, 0.01,
            f"⚠️ Posible colapso: cluster {counts.argmax()} tiene el {max_frac*100:.1f}% de las muestras",
            ha="center", va="bottom", fontsize=9,
            color=ACCENT_KL, alpha=0.9,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.success(f"🌏 Embeddings {method_label} guardados → {save_path}")
