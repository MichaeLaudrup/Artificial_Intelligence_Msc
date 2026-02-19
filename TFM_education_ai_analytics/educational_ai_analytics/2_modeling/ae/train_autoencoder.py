import os
import warnings

# Silence Protobuf and TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
import typer
from sklearn.cluster import KMeans

from educational_ai_analytics.config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    W_WINDOWS,
)

from .hyperparams import AE_PARAMS
from .autoencoder import StudentProfileAutoencoder

app = typer.Typer()


def _set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU Activa: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error configuración GPU: {e}")


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


@app.command()
def main(
    pretrain_epochs: int = AE_PARAMS.pretrain_epochs,
    joint_epochs: int = AE_PARAMS.joint_epochs,
    batch_size: int = AE_PARAMS.batch_size,
    seed: int = 42,
    update_interval: int = 10,    # 🔥 menos oscilación al recalcular P
    sample_frac: float = AE_PARAMS.sample_frac,     # 🔥 compute P on subset (0.3–0.7 recommended)
    target_blend: float = 0.35,   # 0->usar Q (suave), 1->DEC puro (más agresivo)
    shuffle_buf: int = 10000,
    use_mixed_precision: bool = True,
    save_best: bool = True,
):
    """
    Entrenamiento DCN/DEC:
    1) Pretrain: reconstrucción
    2) Init: KMeans en embeddings
    3) Joint: reconstrucción + KL(P||Q)
       Optimizado: update_interval mayor + P sobre subset + tf.function + prefetch + mixed precision
    """
    _configure_gpu()
    _set_seed(seed)

    if use_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        logger.info("⚡ Mixed precision: ACTIVADO (mixed_float16)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    windows = sorted([int(w) for w in W_WINDOWS])

    target_blend = float(np.clip(target_blend, 0.0, 1.0))

    logger.info(
        f"🧠 DCN Training | Ventanas: {windows} | K={AE_PARAMS.n_clusters} | "
        f"batch={batch_size} | update_interval={update_interval} | target_blend={target_blend:.2f}"
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

    logger.info(
        f"📦 X_train: {X_train.shape} | X_val: {None if X_val is None else X_val.shape} "
        f"| N_train={len(X_train)} | N_val={0 if X_val is None else len(X_val)}"
    )

    # -----------------------
    # 2) Phase 1: Pretrain (reconstruction)
    # -----------------------
    logger.info("🚀 Fase 1: Pre-entrenamiento (Reconstrucción)...")
    model = StudentProfileAutoencoder(
        input_dim=X_train.shape[1],
        n_clusters=AE_PARAMS.n_clusters,
    )

    # Subclassed model -> list of losses is robust
    model.compile(
        optimizer=tf.keras.optimizers.Adam(AE_PARAMS.learning_rate),
        loss=[tf.keras.losses.Huber(), tf.keras.losses.MeanSquaredError()],
        loss_weights=[1.0, 0.0],
    )

    y_dummy_train = np.zeros((len(X_train), AE_PARAMS.n_clusters), dtype=np.float32)
    if X_val is not None:
        y_dummy_val = np.zeros((len(X_val), AE_PARAMS.n_clusters), dtype=np.float32)
        val_data = (X_val, [X_val, y_dummy_val])
    else:
        val_data = None

    model.fit(
        X_train,
        [X_train, y_dummy_train],
        validation_data=val_data,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # -----------------------
    # 3) Phase 2: KMeans init
    # -----------------------
    logger.info("📍 Fase 2: Inicializando centroides con KMeans...")
    z = model.get_embeddings(X_train, batch_size=max(1024, batch_size))
    kmeans = KMeans(n_clusters=AE_PARAMS.n_clusters, n_init=20, random_state=seed)
    y_pred = kmeans.fit_predict(z.numpy())

    model.get_layer("clustering_output").set_weights([kmeans.cluster_centers_])
    logger.success("✅ Centroides cargados en la capa de clustering_output.")

    # -----------------------
    # 4) Phase 3: Joint training (optimized)
    # -----------------------
    logger.info("🧬 Fase 3: Optimización Conjunta (DCN) [OPTIMIZADA]...")

    optimizer = tf.keras.optimizers.Adam(AE_PARAMS.learning_rate / 5.0)
    loss_recon = tf.keras.losses.Huber()
    loss_kl = tf.keras.losses.KLDivergence()

    # Best checkpoint tracking (phase 3)
    best_obj = float("inf")
    best_path = MODELS_DIR / "ae_best_global.keras"
    last_path = MODELS_DIR / "ae_last_global.keras"

    @tf.function
    def train_step(x_batch, p_batch):
        with tf.GradientTape() as tape:
            x_rec, q_batch = model(x_batch, training=True)

            # guardrails for KL
            q_batch = tf.clip_by_value(q_batch, 1e-12, 1.0)
            p_batch = tf.clip_by_value(p_batch, 1e-12, 1.0)

            l_rec = loss_recon(x_batch, x_rec)
            l_cl = loss_kl(p_batch, q_batch)

            w_cl = tf.cast(AE_PARAMS.clustering_loss_weight, l_rec.dtype)
            l_aux = tf.add_n(model.losses) if model.losses else tf.zeros_like(l_rec)
            total = l_rec + w_cl * l_cl + l_aux

        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_rec, l_cl, l_aux

    train_p_ds = None
    # Early Stopping tracking
    patience_counter = 0
    patience_limit = AE_PARAMS.early_stopping_patience

    for epoch in range(joint_epochs):
        # Recompute P (target distribution) every update_interval epochs
        if epoch % update_interval == 0:
            logger.info(f"   🔄 Actualizando distribución objetivo (Epoch {epoch})...")

            m = int(len(X_train) * float(sample_frac))
            m = max(m, AE_PARAMS.n_clusters * 50)  # small safety floor
            m = min(m, len(X_train))

            idx = np.random.choice(len(X_train), m, replace=False)
            X_p = X_train[idx]

            # Fast inference for q on subset
            _, q = model.predict(X_p, batch_size=batch_size, verbose=0)
            q = q.astype(np.float32)
            p_hard = target_distribution(q)
            # suaviza el salto tras cada refresh de P para estabilizar KL
            p = ((1.0 - target_blend) * q) + (target_blend * p_hard)
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            p = p.astype(np.float32)

            # label-change diagnostic
            y_curr = q.argmax(1)
            y_prev_subset = y_pred[idx] if len(y_pred) == len(X_train) else y_pred[:m]
            delta_label = float(np.mean(y_curr != y_prev_subset)) * 100.0
            if len(y_pred) == len(X_train):
                y_pred[idx] = y_curr

            logger.info(f"   📊 Cambio en etiquetas (subset): {delta_label:.4f}% | subset={m}")

            train_p_ds = (
                tf.data.Dataset.from_tensor_slices((X_p, p))
                .shuffle(shuffle_buf)
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

        if train_p_ds is None:
            raise RuntimeError("train_p_ds no inicializado.")

        epoch_t = tf.keras.metrics.Mean()
        epoch_r = tf.keras.metrics.Mean()
        epoch_c = tf.keras.metrics.Mean()
        epoch_a = tf.keras.metrics.Mean()

        from tqdm import tqdm
        pbar = tqdm(train_p_ds, desc=f"   Epoch {epoch+1:02d}/{joint_epochs}", leave=False)
        for x_batch, p_batch in pbar:
            total, l_rec, l_cl, l_aux = train_step(x_batch, p_batch)
            epoch_t.update_state(total)
            epoch_r.update_state(l_rec)
            epoch_c.update_state(l_cl)
            epoch_a.update_state(l_aux)
            
            pbar.set_postfix({"T": f"{total.numpy():.4f}", "R": f"{l_rec.numpy():.4f}", "C": f"{l_cl.numpy():.4f}"})

        t_val = float(epoch_t.result().numpy())
        r_val = float(epoch_r.result().numpy())
        c_val = float(epoch_c.result().numpy())
        a_val = float(epoch_a.result().numpy())
        obj_val = r_val + float(AE_PARAMS.clustering_loss_weight) * c_val

        logger.info(
            f"   📉 Finalizado Epoch {epoch+1:02d}/{joint_epochs} | "
            f"T: {t_val:.4f} | Obj(R+λC): {obj_val:.4f} | R: {r_val:.4f} | C: {c_val:.4f} | Aux: {a_val:.4f}"
        )

        # Save best checkpoint & Early Stopping logic
        if save_best and obj_val < best_obj:
            best_obj = obj_val
            patience_counter = 0  # Reset patience
            model.save(best_path)
            logger.info(f"   💾 Guardado BEST (Phase 3) | Obj(R+λC): {best_obj:.4f} -> {best_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.warning(f"   🛑 Early Stopping activado! No hay mejora en {patience_limit} épocas.")
                break

    # Save last state
    model.save(last_path)
    logger.success(f"✨ DCN completado. BEST: {best_path} | LAST: {last_path}")


if __name__ == "__main__":
    app()
