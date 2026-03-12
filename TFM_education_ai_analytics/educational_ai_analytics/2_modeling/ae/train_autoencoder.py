import os
import sys
import warnings
from contextlib import contextmanager

try:
    from .hyperparams import AE_PARAMS
except ImportError:
    from hyperparams import AE_PARAMS

from educational_ai_analytics.tf_runtime import configure_tensorflow_runtime, resolve_execution_device

EXECUTION_DEVICE = resolve_execution_device(AE_PARAMS.execution_device)

# Silence Protobuf and TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Force modern Keras and avoid legacy tf_keras mismatches.
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_CACHE_DISABLE"] = "1"
if EXECUTION_DEVICE == "cpu":
    # Keep this before importing TensorFlow to avoid initializing CUDA in CPU mode.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Render a single in-place progress bar only when stdout/stderr are true TTYs.
IS_TTY = sys.stdout.isatty() and sys.stderr.isatty()

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from educational_ai_analytics.config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    AE_MODELS_DIR,
    AE_REPORTS_DIR,
    W_WINDOWS,
)
from .autoencoder import StudentProfileAutoencoder
from .training_reporter import TrainingMetricsCollector, plot_training_evolution, plot_embeddings_pca

app = typer.Typer()


class _ValLossOnlyCallback(tf.keras.callbacks.Callback):
    """Print only val_loss each epoch to keep terminal output clean."""

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            logger.info(f"Epoch {epoch + 1}/{self.total_epochs} | val_loss: N/A")
        else:
            logger.info(f"Epoch {epoch + 1}/{self.total_epochs} | val_loss: {val_loss:.4f}")


class _FilteredStderr:
    """Drop known noisy low-level CUDA/XLA lines without hiding real errors."""

    _DROP_TOKENS = (
        "could not open file to read NUMA node",
        "Your kernel may have been built without NUMA support.",
        "XLA service",
        "Compiled cluster using XLA",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, data):
        for line in data.splitlines(True):
            if any(token in line for token in self._DROP_TOKENS):
                continue
            self._wrapped.write(line)

    def flush(self):
        self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()


@contextmanager
def _suppress_low_level_cuda_noise():
    original_stderr = sys.stderr
    sys.stderr = _FilteredStderr(original_stderr)
    try:
        yield
    finally:
        sys.stderr = original_stderr


def _set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_gpu():
    return configure_tensorflow_runtime(tf, EXECUTION_DEVICE, logger)


def load_features(path: Path, W: int):
    X0_path = path / "day0_static_features.csv"
    Xdyn_path = path / "ae_uptow_features" / f"ae_uptow_features_w{W:02d}.csv"

    if not X0_path.exists() or not Xdyn_path.exists():
        raise FileNotFoundError(f"Faltan features en {path} para W{W}")

    df0 = pd.read_csv(X0_path, index_col=0)
    dfdyn = pd.read_csv(Xdyn_path, index_col=0)

    df = pd.concat([df0, dfdyn.reindex(df0.index)], axis=1).fillna(0.0)
    X = df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
    return X


def target_distribution(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    DEC target distribution P (sharpening), stabilized.
    q: (N, K)
    """
    q = np.clip(q, eps, 1.0)
    weight = (q ** 2) / (q.sum(axis=0, keepdims=True) + eps)
    p = weight / (weight.sum(axis=1, keepdims=True) + eps)
    return p.astype(np.float32)


def _compute_clustering_quality(
    z_val: np.ndarray,
    labels_val: np.ndarray,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calcula Silhouette y Davies-Bouldin en validación (con muestreo opcional)."""
    if z_val.size == 0 or labels_val.size == 0:
        return float("nan"), float("nan")

    n = z_val.shape[0]
    if n < 3:
        return float("nan"), float("nan")

    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        z_eval = z_val[idx]
        labels_eval = labels_val[idx]
    else:
        z_eval = z_val
        labels_eval = labels_val

    n_clusters_present = np.unique(labels_eval).size
    if n_clusters_present < 2 or n_clusters_present >= len(labels_eval):
        return float("nan"), float("nan")

    try:
        sil = float(silhouette_score(z_eval, labels_eval, metric="euclidean"))
        db = float(davies_bouldin_score(z_eval, labels_eval))
        return sil, db
    except Exception:
        return float("nan"), float("nan")


@app.command()
def main(
    pretrain_epochs: int = AE_PARAMS.pretrain_epochs,
    joint_epochs: int = AE_PARAMS.joint_epochs,
    batch_size: int = AE_PARAMS.batch_size,
    seed: int = 42,
    update_interval: int = 10,
    sample_frac: float = AE_PARAMS.sample_frac,
    target_blend: float = AE_PARAMS.target_blend,
    shuffle_buf: int = 10000,
    use_mixed_precision: bool = AE_PARAMS.use_mixed_precision,
    save_best: bool = True,
    use_clustering_objective: bool = AE_PARAMS.use_clustering_objective,
    clustering_loss_scale: float = AE_PARAMS.clustering_loss_scale,
    # 🆕 Early stopping diferido: la paciencia no empieza hasta que el warmup termina.
    # Durante el warmup, ValObj sube por diseño (blend más alto = P más dura).
    # Monitorizamos val_recon porque es estable y representa el verdadero riesgo
    # de colapso del espacio latente.
    early_stop_start_epoch: int = AE_PARAMS.target_blend_warmup_epochs,
):
    """
    Entrenamiento DCN/DEC:
    1) Pretrain: reconstrucción
    2) Init: KMeans en embeddings
    3) Joint: reconstrucción + KL(P||Q)
    Al finalizar guarda reports/ae/training_evolution.png con gráficas ilustrativas.
    """
    runtime_device = _configure_gpu()
    _set_seed(seed)

    AE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_mixed_precision and runtime_device == "gpu":
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        logger.info("⚡ Mixed precision: ACTIVADO (mixed_float16)")
    elif use_mixed_precision and runtime_device == "cpu":
        use_mixed_precision = False
        logger.info("⚡ Mixed precision: DESACTIVADO (sin beneficio en CPU)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    windows = sorted([int(w) for w in W_WINDOWS])
    target_blend = float(np.clip(target_blend, 0.0, 1.0))

    logger.info(
        f"🧠 DCN Training | Ventanas: {windows} | K={AE_PARAMS.n_clusters} | "
        f"batch={batch_size} | update_interval={update_interval} | target_blend={target_blend:.2f} | "
        f"use_clustering_objective={use_clustering_objective} | λ={AE_PARAMS.clustering_loss_weight} | scale={clustering_loss_scale}"
    )

    # -----------------------
    # 1) Load data
    # -----------------------
    X_train_list, X_val_list = [], []
    for W in windows:
        try:
            X_train_list.append(load_features(FEATURES_DATA_DIR / "training", W))
            X_val_list.append(load_features(FEATURES_DATA_DIR / "validation", W))
        except Exception as e:
            logger.warning(f"   ⚠️ W{W:02d}: {e}")

    if not X_train_list:
        raise RuntimeError("No se pudieron cargar features de training en ninguna ventana.")

    X_train = np.vstack(X_train_list).astype(np.float32)
    X_val = np.vstack(X_val_list).astype(np.float32) if X_val_list else None

    logger.info(f"📦 X_train: {X_train.shape} | X_val: {None if X_val is None else X_val.shape}")

    # -----------------------
    # 2) Phase 1: Pretrain (reconstruction)
    # -----------------------
    logger.info("🚀 Fase 1: Pre-entrenamiento (Reconstrucción)...")
    model = StudentProfileAutoencoder(
        input_dim=X_train.shape[1],
        n_clusters=AE_PARAMS.n_clusters,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(AE_PARAMS.learning_rate),
        loss=[tf.keras.losses.Huber(), tf.keras.losses.MeanSquaredError()],
        loss_weights=[1.0, 0.0],
        jit_compile=False,
    )

    y_dummy_train = np.zeros((len(X_train), AE_PARAMS.n_clusters), dtype=np.float32)
    if X_val is not None:
        y_dummy_val = np.zeros((len(X_val), AE_PARAMS.n_clusters), dtype=np.float32)
        val_data = (X_val, [X_val, y_dummy_val])
    else:
        val_data = None

    with _suppress_low_level_cuda_noise():
        pretrain_hist = model.fit(
            X_train,
            [X_train, y_dummy_train],
            validation_data=val_data,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[_ValLossOnlyCallback(pretrain_epochs)],
        )

    # -----------------------
    # 3) Phase 2: KMeans init
    # -----------------------
    if use_clustering_objective:
        logger.info("📍 Fase 2: Inicializando centroides con KMeans...")
        z = model.get_embeddings(X_train, batch_size=max(1024, batch_size))
        kmeans = KMeans(n_clusters=AE_PARAMS.n_clusters, n_init=20, random_state=seed)
        y_pred = kmeans.fit_predict(z.numpy())
        model.get_layer("clustering_output").set_weights([kmeans.cluster_centers_])
        logger.success("✅ Centroides cargados en la capa de clustering_output.")
    else:
        y_pred = np.zeros((len(X_train),), dtype=np.int32)
        logger.warning("⏭️ Clustering objetivo DESACTIVADO: se omiten KMeans init y pérdida KL.")

    # -----------------------
    # 4) Phase 3: Joint training
    # -----------------------
    if use_clustering_objective:
        logger.info("🧬 Fase 3: Optimización Conjunta (DCN) [OPTIMIZADA]...")
    else:
        logger.info("🧬 Fase 3: Fine-tuning SOLO reconstrucción (ablation limpia)...")

    _base_lr = AE_PARAMS.learning_rate / AE_PARAMS.lr_phase3_divisor
    _base_opt = tf.keras.optimizers.Adam(_base_lr)
    # ✅ FIX #1 — LossScaleOptimizer previene underflow de gradientes con mixed_float16.
    # En custom loops sin esto, los gradientes de losses pequeños (Huber ~0.01, KL ~0.001)
    # se redondean a cero en float16 silenciosamente, degradando el entrenamiento.
    if use_mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        optimizer = mp.LossScaleOptimizer(_base_opt)
        logger.info(f"   ⚡ LR fase 3: {_base_lr:.2e} (LR/{AE_PARAMS.lr_phase3_divisor:.0f}) "
                    f"| grad_clip={AE_PARAMS.grad_clip_norm} | LossScaleOptimizer: ACTIVO")
    else:
        optimizer = _base_opt
        logger.info(f"   ⚡ LR fase 3: {_base_lr:.2e} (LR/{AE_PARAMS.lr_phase3_divisor:.0f}) "
                    f"| grad_clip={AE_PARAMS.grad_clip_norm}")
    loss_recon = tf.keras.losses.Huber()
    loss_kl = tf.keras.losses.KLDivergence()

    best_obj = float("inf")
    
    # Rutas para checkpoints por época
    checkpoint_dir = AE_MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = AE_MODELS_DIR / "ae_best_global.keras"
    last_path = AE_MODELS_DIR / "ae_last_global.keras"

    metrics = TrainingMetricsCollector()

    # ─── Bloque de losses compartido (siempre float32) ────────────────────────
    @tf.autograph.experimental.do_not_convert
    def _compute_loss(x_batch, p_batch, is_training: bool):
        x_rec, q_batch = model(x_batch, training=is_training)
        x32      = tf.cast(x_batch, tf.float32)
        xrec32   = tf.cast(x_rec,   tf.float32)
        q32      = tf.clip_by_value(tf.cast(q_batch, tf.float32), 1e-12, 1.0)
        p32      = tf.clip_by_value(tf.cast(p_batch, tf.float32), 1e-12, 1.0)
        l_rec    = loss_recon(x32, xrec32)
        l_cl_raw = loss_kl(p32, q32)
        l_cl     = l_cl_raw * tf.cast(clustering_loss_scale, tf.float32)
        w_cl     = tf.constant(
            AE_PARAMS.clustering_loss_weight if use_clustering_objective else 0.0,
            dtype=tf.float32,
        )
        l_aux = (
            tf.cast(tf.add_n(model.losses), tf.float32)
            if model.losses
            else tf.zeros((), dtype=tf.float32)
        )
        total = l_rec + w_cl * l_cl + l_aux
        return total, l_rec, l_cl_raw, l_cl, l_aux

    # ─── Mixed precision: scale_loss() → gradientes escalados → unscale ──────
    @tf.function(jit_compile=False, reduce_retracing=True)
    def _step_mp(x_batch, p_batch):
        with tf.GradientTape() as tape:
            total, l_rec, l_cl_raw, l_cl, l_aux = _compute_loss(
                x_batch, p_batch, is_training=True
            )
            # scale_loss multiplica total por el factor de escala actual (tensor TF)
            scaled = optimizer.scale_loss(total)
        raw_grads = tape.gradient(scaled, model.trainable_variables)
        # Unscale: el factor = scaled / total (ambos son tensores → trazable por autograph)
        # Usar safe_div para evitar div/0 si total es exactamente 0
        scale_factor = tf.math.divide_no_nan(scaled, total)
        inv = tf.math.reciprocal(tf.maximum(scale_factor, 1e-7))
        grads = [g * inv if g is not None else g for g in raw_grads]
        grads, _ = tf.clip_by_global_norm(grads, AE_PARAMS.grad_clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_rec, l_cl_raw, l_cl, l_aux

    # ─── Float32 puro: sin escala ─────────────────────────────────────────────
    @tf.function(jit_compile=False, reduce_retracing=True)
    def _step_fp32(x_batch, p_batch):
        with tf.GradientTape() as tape:
            total, l_rec, l_cl_raw, l_cl, l_aux = _compute_loss(
                x_batch, p_batch, is_training=True
            )
        grads = tape.gradient(total, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, AE_PARAMS.grad_clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_rec, l_cl_raw, l_cl, l_aux

    # Dispatch Python puro — ningún condicional entra al grafo TF.
    # En algunas combinaciones TF/Keras, LossScaleOptimizer no expone scale_loss.
    can_scale_loss = bool(use_mixed_precision and hasattr(optimizer, "scale_loss"))
    if use_mixed_precision and not can_scale_loss:
        logger.warning("   ⚠️ mixed_precision activa, pero optimizer.scale_loss no está disponible; se usará paso FP32 estable.")
    train_step = _step_mp if can_scale_loss else _step_fp32

    train_p_ds = None
    patience_counter = 0
    patience_limit = AE_PARAMS.early_stopping_patience

    from tqdm import tqdm
    epoch_pbar = tqdm(
        total=joint_epochs,
        desc="   Phase3",
        leave=False,
        disable=not IS_TTY,
        dynamic_ncols=True,
    )

    for epoch in range(joint_epochs):
        # 🔧 FIX #3 — warmup lineal de target_blend: empieza suave y sube gradualmente
        if use_clustering_objective:
            warmup_progress = min(1.0, epoch / max(1, AE_PARAMS.target_blend_warmup_epochs))
            current_blend = AE_PARAMS.target_blend + warmup_progress * (
                AE_PARAMS.target_blend_max - AE_PARAMS.target_blend
            )
        else:
            current_blend = 0.0
        # Recompute P every update_interval epochs
        if use_clustering_objective and epoch % update_interval == 0:
            logger.info(f"   🔄 Actualizando distribución objetivo (Epoch {epoch} | blend={current_blend:.3f})...")

            m = int(len(X_train) * float(sample_frac))
            m = max(m, AE_PARAMS.n_clusters * 50)
            m = min(m, len(X_train))
            idx = np.random.choice(len(X_train), m, replace=False)

            # ✅ FIX #3 — P se calcula SOLO sobre X_train[idx] (sin leakage de val).
            # Antes se incluía X_val para que los centroides "vieran" el espacio de val,
            # pero eso es leakage suave: val influye en el target P que guía el training,
            # haciendo que la métrica de val sea menos fiable como estimador real.
            _, q_train_p = model.predict(X_train[idx], batch_size=batch_size, verbose=0)
            q_train_p = q_train_p.astype(np.float32)
            p_hard = target_distribution(q_train_p)
            p_train_only = ((1.0 - current_blend) * q_train_p) + (current_blend * p_hard)
            p_train_only = np.clip(p_train_only, 1e-12, 1.0)
            p_train_only = p_train_only / p_train_only.sum(axis=1, keepdims=True)

            # label-change diagnostic
            y_curr = q_train_p.argmax(1)
            y_prev_subset = y_pred[idx] if len(y_pred) == len(X_train) else y_pred[:m]
            delta_label = float(np.mean(y_curr != y_prev_subset)) * 100.0
            if len(y_pred) == len(X_train):
                y_pred[idx] = y_curr

            logger.info(f"   📊 Cambio en etiquetas (subset): {delta_label:.4f}% | subset={m} | blend={current_blend:.3f}")

            train_p_ds = (
                tf.data.Dataset.from_tensor_slices((X_train[idx], p_train_only))
                .shuffle(shuffle_buf)
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

        if not use_clustering_objective and epoch == 0:
            p_train = np.zeros((len(X_train), AE_PARAMS.n_clusters), dtype=np.float32)
            current_blend = 0.0
            train_p_ds = (
                tf.data.Dataset.from_tensor_slices((X_train, p_train))
                .shuffle(shuffle_buf)
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

        if train_p_ds is None:
            raise RuntimeError("train_p_ds no inicializado.")

        epoch_t = tf.keras.metrics.Mean()
        epoch_r = tf.keras.metrics.Mean()
        epoch_c_raw = tf.keras.metrics.Mean()
        epoch_c = tf.keras.metrics.Mean()
        epoch_a = tf.keras.metrics.Mean()

        with _suppress_low_level_cuda_noise():
            for x_batch, p_batch in train_p_ds:
                total, l_rec, l_cl_raw, l_cl, l_aux = train_step(x_batch, p_batch)
                epoch_t.update_state(total)
                epoch_r.update_state(l_rec)
                epoch_c_raw.update_state(l_cl_raw)
                epoch_c.update_state(l_cl)
                epoch_a.update_state(l_aux)

        train_t_val     = float(epoch_t.result().numpy())
        train_r_val     = float(epoch_r.result().numpy())
        train_c_raw_val = float(epoch_c_raw.result().numpy())
        train_c_val     = float(epoch_c.result().numpy())
        train_a_val     = float(epoch_a.result().numpy())
        lambda_cl       = float(AE_PARAMS.clustering_loss_weight if use_clustering_objective else 0.0)
        train_cl_contrib = lambda_cl * train_c_val
        train_obj_val    = train_r_val + train_cl_contrib
        train_cl_share_pct = (100.0 * train_cl_contrib / max(train_obj_val, 1e-12)) if train_obj_val > 0 else 0.0

        if X_val is not None:
            x_rec_val, q_val = model.predict(X_val, batch_size=batch_size, verbose=0)
            val_r = float(np.mean(tf.keras.losses.huber(X_val, x_rec_val).numpy()))
            val_sil = float("nan")
            val_db = float("nan")
            if use_clustering_objective:
                q_val = np.clip(q_val.astype(np.float32), 1e-12, 1.0)
                p_val_hard = target_distribution(q_val)
                # 🐛 FIX #4 — usar current_blend (no target_blend fijo).
                # Con target_blend=0.05 (inicial), P_val ≈ Q_val → KL(P||Q) ≈ 0
                # haciendo que val_kl_raw aparezca plano en la gráfica.
                # Ahora usamos el mismo blend progresivo que en training.
                p_val = ((1.0 - current_blend) * q_val) + (current_blend * p_val_hard)
                p_val = np.clip(p_val, 1e-12, 1.0)
                p_val = p_val / p_val.sum(axis=1, keepdims=True)
                val_c_raw = float(np.mean(
                    tf.keras.losses.kullback_leibler_divergence(p_val, q_val).numpy()
                ))
                val_c = val_c_raw * float(clustering_loss_scale)
                # Diagnóstico: aviso si val_kl sigue siendo sospechosamente bajo
                if val_c_raw < 1e-6:
                    logger.warning(
                        f"   ⚠️  Epoch {epoch+1}: val_kl_raw={val_c_raw:.2e} sospechosamente bajo. "
                        f"blend={current_blend:.3f} | q_val min={q_val.min():.4f} max={q_val.max():.4f}"
                    )

                # Métricas de calidad de clustering en espacio latente (validación)
                try:
                    z_val = model.get_embeddings(X_val, batch_size=max(1024, batch_size)).numpy()
                    labels_val = q_val.argmax(axis=1)
                    val_sil, val_db = _compute_clustering_quality(
                        z_val=z_val,
                        labels_val=labels_val,
                        max_samples=5000,
                        seed=seed + epoch,
                    )
                except Exception as e:
                    logger.debug(f"   ℹ️  No se pudieron calcular métricas de clustering en epoch {epoch+1}: {e}")
            else:
                val_c_raw = 0.0
                val_c = 0.0
                val_sil = float("nan")
                val_db = float("nan")
            val_cl_contrib = lambda_cl * val_c
            model_obj = val_r + val_cl_contrib
            monitor_name = "ValObj"
        else:
            val_r = val_c_raw = val_c = val_cl_contrib = 0.0
            val_sil = val_db = float("nan")
            model_obj = train_obj_val
            monitor_name = "TrainObj"

        logger.info(
            f"   📉 Finalizado Epoch {epoch+1:02d}/{joint_epochs} | "
            f"Train T: {train_t_val:.4f} | Train Obj(R+λC): {train_obj_val:.4f} | Train R: {train_r_val:.4f} | "
            f"Train Craw: {train_c_raw_val:.4f} | Train Cscaled: {train_c_val:.4f} | "
            f"Train λ·C: {train_cl_contrib:.4f} ({train_cl_share_pct:.2f}%) | Aux: {train_a_val:.4f} | "
            f"Val R: {val_r:.4f} | Val Craw: {val_c_raw:.4f} | Val Cscaled: {val_c:.4f} | Val λ·C: {val_cl_contrib:.4f} | "
            f"Sil: {val_sil if np.isfinite(val_sil) else float('nan'):.4f} | "
            f"DB: {val_db if np.isfinite(val_db) else float('nan'):.4f} | "
            f"{monitor_name}: {model_obj:.4f}"
        )

        if IS_TTY:
            epoch_pbar.set_postfix({
                "epoch": f"{epoch+1}/{joint_epochs}",
                "TrainObj": f"{train_obj_val:.4f}",
                "ValR": f"{val_r:.4f}",
            })
        epoch_pbar.update(1)

        # Record metrics for report
        metrics.record(
            epoch=epoch + 1,
            train_recon=train_r_val,
            train_kl_raw=train_c_raw_val,
            train_kl_scaled=train_c_val,
            train_obj=train_obj_val,
            val_recon=val_r,
            val_kl_raw=val_c_raw,
            model_obj=model_obj,
            val_silhouette=val_sil,
            val_davies=val_db,
        )

        # ─── Diagnósticos baratos de clustering ───────────────────────────────
        # Se calculan sobre q_val ya computada. Costo: O(N·K) en CPU, insignificante.
        if use_clustering_objective and X_val is not None:
            q_diag = np.clip(q_val, 1e-12, 1.0)     # q_val ya calculado arriba
            # 1) Entropía media: si baja demasiado rápido → asignaciones se vuelven duras
            entropy_mean = float(-np.mean(np.sum(q_diag * np.log(q_diag), axis=1)))
            # 2) Fracción del cluster más grande: >0.9 → posible colapso a un solo cluster
            cluster_counts = np.bincount(q_diag.argmax(axis=1), minlength=AE_PARAMS.n_clusters)
            max_cluster_frac = float(cluster_counts.max() / len(q_diag))
            logger.info(
                f"   🔬 Diagnóstico Q | Entropía media: {entropy_mean:.4f} "
                f"| Cluster más grande: {max_cluster_frac*100:.1f}% "
                f"| Tamaños: {cluster_counts.tolist()}"
            )
            if max_cluster_frac > 0.80:
                logger.warning(f"   ⚠️  Posible colapso de cluster: {max_cluster_frac*100:.1f}% en un solo cluster")

        # ─── Checkpoint y Early Stopping ─────────────────────────────────────
        # Monitor: val_recon (no ValObj). El KL sube por diseño durante warmup;
        # usar ValObj lleva a early stopping prematuro. La reconstrucción es la
        # señal que indica colapso real del espacio latente.
        # La paciencia solo empieza a contar después del fin del warmup.
        monitor_val = val_r   # val_recon como criterio de parada
        in_warmup   = (epoch + 1) <= early_stop_start_epoch

        # Siempre guardamos el checkpoint de la época actual si mejora el val_recon 
        # para evitar llenar el disco, pero guardamos el history completo
        ckpt_path = checkpoint_dir / f"ae_epoch_{epoch+1:02d}.keras"
        model.save(ckpt_path)

        # Mantenemos val_recon como monitor estricto del Early Stopping (para evitar colapso)
        if save_best and monitor_val < best_obj:
            best_obj = monitor_val
            patience_counter = 0
            logger.info(f"   💾 Nuevo mínimo de val_recon: {best_obj:.6f} (Epoch {epoch+1})")
        elif not in_warmup:  # solo penalizar paciencia fuera del warmup
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(
                    f"   🛑 Early Stopping activado (post-warmup) "
                    f"| sin mejora en val_recon durante {patience_limit} épocas."
                )
                break
        else:
            logger.debug(f"   ⏳ En warmup (epoch {epoch+1}/{early_stop_start_epoch}) — paciencia suspendida")

    epoch_pbar.close()

    model.save(last_path)
    logger.success(f"✨ Entrenamiento terminado. Procediendo a seleccionar el mejor modelo...")

    # --- Selección Híbrida del Mejor Modelo (sincronizada con la gráfica) ---
    final_selected_epoch = metrics.selected_epoch
    if final_selected_epoch is not None:
        target_ckpt = checkpoint_dir / f"ae_epoch_{final_selected_epoch:02d}.keras"
        if target_ckpt.exists():
            import shutil
            shutil.copy(target_ckpt, best_path)
            logger.success(f"🏆 Modelo Final Seleccionado: Época {final_selected_epoch} -> {best_path.name}")
            logger.info(f"   Razonamiento: seleccion_híbrida (val_recon dentro del 5% del mínimo + maximiza Silhouette/DB).")
        else:
            logger.warning(f"⚠️ Checkpoint para época {final_selected_epoch} no encontrado. Se mantendrá el último como fallback.")
    else:
        logger.warning(f"⚠️ No se pudo determinar selected_epoch (falta de datos).")

    # Limpieza de checkpoints intermedios
    logger.info("🧹 Limpiando checkpoints temporales...")
    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    logger.success(f"✨ DCN completado. BEST: {best_path.name} | LAST: {last_path.name}")

    # -----------------------
    # 5) Save training plots
    # -----------------------
    plot_training_evolution(
        pretrain_history=pretrain_hist.history,
        collector=metrics,
        use_clustering_objective=use_clustering_objective,
        save_path=AE_REPORTS_DIR / "training_evolution.png",
    )

    # -----------------------
    # 6) PCA de embeddings
    # -----------------------
    logger.info("📉 Generando visualización PCA del espacio latente...")
    try:
        # Cargar el mejor modelo guardado
        best_model = tf.keras.models.load_model(
            best_path,
            custom_objects={"StudentProfileAutoencoder": StudentProfileAutoencoder},
            compile=False
        )

        # Concatenar train + val para ver el espacio latente completo
        if X_val is not None:
            X_all = np.vstack([X_train, X_val])
        else:
            X_all = X_train

        # Obtener embeddings y asignaciones de cluster
        z_all = best_model.get_embeddings(X_all, batch_size=max(1024, batch_size)).numpy()
        _, q_all = best_model.predict(X_all, batch_size=max(1024, batch_size), verbose=0)
        cluster_labels_all = q_all.argmax(axis=1)

        plot_embeddings_pca(
            embeddings=z_all,
            cluster_labels=cluster_labels_all,
            save_path=AE_REPORTS_DIR / "embeddings_pca.png",
            n_clusters=AE_PARAMS.n_clusters,
            title_suffix="Best model",
        )
    except Exception as e:
        logger.warning(f"   ⚠️ No se pudo generar PCA: {e}")



if __name__ == "__main__":
    app()
