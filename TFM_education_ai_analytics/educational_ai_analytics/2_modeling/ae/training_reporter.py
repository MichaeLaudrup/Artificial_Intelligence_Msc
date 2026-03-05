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
    val_silhouette:  list[float] = field(default_factory=list)
    val_davies:      list[float] = field(default_factory=list)

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
        val_silhouette: float = np.nan,
        val_davies: float = np.nan,
    ) -> None:
        self.epochs.append(epoch)
        self.train_recon.append(train_recon)
        self.train_kl_raw.append(train_kl_raw)
        self.train_kl_scaled.append(train_kl_scaled)
        self.train_obj.append(train_obj)
        self.val_recon.append(val_recon)
        self.val_kl_raw.append(val_kl_raw)
        self.model_obj.append(model_obj)
        self.val_silhouette.append(val_silhouette)
        self.val_davies.append(val_davies)

    @property
    def best_epoch(self) -> Optional[int]:
        """Época (1-indexed) con el menor ``model_obj``."""
        return self.epochs[int(np.argmin(self.model_obj))] if self.model_obj else None

    @property
    def best_epoch_val_recon(self) -> Optional[int]:
        """Época (1-indexed) con la menor ``val_recon`` válida (>0)."""
        if not self.epochs or not self.val_recon:
            return None
        val_arr = np.asarray(self.val_recon, dtype=float)
        finite = np.isfinite(val_arr)
        has_val = finite & (val_arr > 0.0)
        if not np.any(has_val):
            return None
        idx = int(np.argmin(val_arr[has_val]))
        return int(np.asarray(self.epochs, dtype=int)[has_val][idx])

    @property
    def last_epoch(self) -> Optional[int]:
        return self.epochs[-1] if self.epochs else None

    @property
    def best_epoch_silhouette(self) -> Optional[int]:
        """Época con mayor silhouette en validación."""
        if not self.epochs or not self.val_silhouette:
            return None
        sil = np.asarray(self.val_silhouette, dtype=float)
        finite = np.isfinite(sil)
        if not np.any(finite):
            return None
        idx = int(np.argmax(sil[finite]))
        return int(np.asarray(self.epochs, dtype=int)[finite][idx])

    @property
    def best_epoch_davies(self) -> Optional[int]:
        """Época con menor Davies-Bouldin en validación."""
        if not self.epochs or not self.val_davies:
            return None
        db = np.asarray(self.val_davies, dtype=float)
        finite = np.isfinite(db) & (db > 0.0)
        if not np.any(finite):
            return None
        idx = int(np.argmin(db[finite]))
        return int(np.asarray(self.epochs, dtype=int)[finite][idx])

    @property
    def selected_epoch(self) -> Optional[int]:
        """
        Época final seleccionada por compromiso estructural:
          1) Restringe a épocas con val_recon <= min(val_recon) * (1 + 1%).
          2) Dentro de ese subconjunto, maximiza score combinado de Silhouette↑ y DB↓.
          3) En empate, elige la época más tardía.
        """
        if not self.epochs:
            return None

        epochs = np.asarray(self.epochs, dtype=int)
        recon = np.asarray(self.val_recon, dtype=float)
        sil = np.asarray(self.val_silhouette, dtype=float) if self.val_silhouette else np.full_like(recon, np.nan)
        db = np.asarray(self.val_davies, dtype=float) if self.val_davies else np.full_like(recon, np.nan)

        recon_ok = np.isfinite(recon) & (recon > 0.0)
        if not np.any(recon_ok):
            return self.best_epoch_val_recon or self.best_epoch

        recon_tol = 0.01
        recon_min = float(np.min(recon[recon_ok]))
        candidates = recon_ok & (recon <= recon_min * (1.0 + recon_tol))
        if not np.any(candidates):
            return self.best_epoch_val_recon or self.best_epoch

        scores = np.full(len(epochs), np.nan, dtype=float)
        metric_count = 0

        sil_ok = candidates & np.isfinite(sil)
        if np.any(sil_ok):
            sil_vals = sil[sil_ok]
            sil_rng = float(np.max(sil_vals) - np.min(sil_vals))
            sil_norm = (sil_vals - np.min(sil_vals)) / (sil_rng + 1e-12)
            scores[sil_ok] = np.nan_to_num(scores[sil_ok], nan=0.0) + sil_norm
            metric_count += 1

        db_ok = candidates & np.isfinite(db) & (db > 0.0)
        if np.any(db_ok):
            db_vals = db[db_ok]
            db_rng = float(np.max(db_vals) - np.min(db_vals))
            db_norm = (np.max(db_vals) - db_vals) / (db_rng + 1e-12)
            scores[db_ok] = np.nan_to_num(scores[db_ok], nan=0.0) + db_norm
            metric_count += 1

        if metric_count == 0:
            return self.best_epoch_val_recon or self.best_epoch

        valid_scores = candidates & np.isfinite(scores)
        if not np.any(valid_scores):
            return self.best_epoch_val_recon or self.best_epoch

        best_score = float(np.max(scores[valid_scores]))
        tied = np.where(valid_scores & (np.abs(scores - best_score) <= 1e-12))[0]
        return int(epochs[tied[-1]]) if tied.size else (self.best_epoch_val_recon or self.best_epoch)


# ─── Utilidades de suavizado ─────────────────────────────────────────────────────

def _ema(values: list[float], alpha: float = 0.3) -> np.ndarray:
    """Exponential Moving Average de una serie."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return arr
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
    if not x or not y:
        return

    n = min(len(x), len(y))
    x_arr = np.asarray(x[:n])
    y_arr = np.asarray(y[:n], dtype=float)
    finite_mask = np.isfinite(y_arr)
    if not np.any(finite_mask):
        return

    x_plot = x_arr[finite_mask]
    y_plot = y_arr[finite_mask]
    if y_plot.size == 0:
        return

    ax.plot(x_plot, y_plot, color=color, linewidth=0.8, alpha=MUTED_ALPHA,
            linestyle=linestyle, marker=None)
    smooth = _ema(y_plot.tolist(), alpha=alpha)
    ax.plot(x_plot, smooth, color=color, linewidth=2.2,
            linestyle=linestyle, label=label,
            marker=marker, markersize=3, markevery=max(1, len(x_plot) // 12))


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
    best_recon_epoch = j.best_epoch_val_recon
    if best_recon_epoch is not None:
        _annotate_epoch(ax, best_recon_epoch,
                        f"Mejor Recon ({best_recon_epoch})", ACCENT_VAL)

    selected_epoch = j.selected_epoch
    if selected_epoch is not None:
        _annotate_epoch(ax, selected_epoch,
                        f"Seleccionada final ({selected_epoch})", ACCENT_SMOOTH, yrel=0.82)

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
        epochs_arr = np.asarray(j.epochs, dtype=float)
        val_kl_arr = np.asarray(j.val_kl_raw, dtype=float)
        finite = np.isfinite(epochs_arr) & np.isfinite(val_kl_arr)
        trend = np.polyfit(epochs_arr[finite], val_kl_arr[finite], 1)[0] if np.sum(finite) >= 2 else 0.0
        if np.isfinite(trend) and trend > 0:
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
    Panel 4 — Calidad de clustering en validación:
      · Silhouette (eje izq., mayor es mejor).
      · Davies-Bouldin (eje der., menor es mejor).
    """
    j = collector
    has_sil = np.isfinite(np.asarray(j.val_silhouette, dtype=float)).any() if j.val_silhouette else False
    has_db = np.isfinite(np.asarray(j.val_davies, dtype=float)).any() if j.val_davies else False
    if not (has_sil or has_db):
        _blank_panel(ax, "Sin métricas de\nclustering por época",
                     "Calidad de Clustering (Val)")
        return

    if has_sil:
        _plot_with_ema(ax, j.epochs, j.val_silhouette,
                       ACCENT_VAL, "Silhouette (val)", marker="o")

    _style_ax(ax,
              "Calidad de Clustering en Validación\n(Silhouette vs Davies-Bouldin)",
              ylabel="Silhouette (↑ mejor)")

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    if has_db:
        _plot_with_ema(ax2, j.epochs, j.val_davies,
                       ACCENT_KL, "Davies-Bouldin (val)", marker="s", linestyle="--")
    ax2.set_ylabel("Davies-Bouldin (↓ mejor)", color=ACCENT_KL, fontsize=8.5)
    ax2.tick_params(colors=ACCENT_KL, labelsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID_COLOR)
    ax2.spines["left"].set_color(GRID_COLOR)
    ax2.spines["bottom"].set_color(GRID_COLOR)

    # Anotación early stopping
    if j.last_epoch is not None:
        ax.axvline(j.last_epoch, color=ACCENT_KL, linestyle=":",
                   linewidth=1.1, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(j.last_epoch - 0.3,
                ylim[0] + 0.88 * (ylim[1] - ylim[0]),
                f"Early Stop\n(ep. {j.last_epoch})",
                color=ACCENT_KL, fontsize=6.5, ha="right", va="top")

    if j.best_epoch_silhouette is not None:
        _annotate_epoch(ax, j.best_epoch_silhouette,
                        f"Mejor Sil ({j.best_epoch_silhouette})", ACCENT_VAL, yrel=0.85)
    if j.best_epoch_davies is not None:
        _annotate_epoch(ax, j.best_epoch_davies,
                        f"Mejor DB ({j.best_epoch_davies})", ACCENT_KL, yrel=0.70)

    selected_epoch = j.selected_epoch
    if selected_epoch is not None:
        _annotate_epoch(ax, selected_epoch,
                        f"Seleccionada final ({selected_epoch})", ACCENT_SMOOTH, yrel=0.55)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)


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
            4. Fase 3 — Calidad de clustering en validación (Silhouette y Davies-Bouldin).

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

    tmp_save_path = save_path.with_name(f"{save_path.stem}.tmp{save_path.suffix}")
    try:
        fig.savefig(tmp_save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        tmp_save_path.replace(save_path)
    finally:
        plt.close(fig)
        if tmp_save_path.exists():
            tmp_save_path.unlink(missing_ok=True)
    logger.success(f"📊 Gráficas guardadas → {save_path}")


# ─── PCA de embeddings ─────────────────────────────────────────────────────────

def plot_embeddings_pca(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path,
    n_clusters: int,
    title_suffix: str = "",
) -> None:
    """
    Genera una figura 2-panel con la proyección 2D de los embeddings:
            - Panel izq.: PCA.
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
    from sklearn.decomposition import PCA
    method_label = "PCA"
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    logger.info("   📉 PCA proyección completada")

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

    # ── Panel izquierdo: scatter PCA ─────────────────────────────
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
        # Centroide del cluster en el espacio PCA
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
    logger.success(f"📉 Embeddings {method_label} guardados → {save_path}")


def plot_embeddings_umap(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path,
    n_clusters: int,
    title_suffix: str = "",
) -> None:
    """Compatibilidad retroactiva: redirige a proyección PCA."""
    plot_embeddings_pca(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        save_path=save_path,
        n_clusters=n_clusters,
        title_suffix=title_suffix,
    )
