import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import typer
import numpy as np
import tensorflow as tf
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from typing import Optional
import copy

from transformer_GLU_classifier import GLUTransformerClassifier
from hyperparams import TRANSFORMER_PARAMS
from educational_ai_analytics.config import W_WINDOWS

app = typer.Typer()

def filter_classes(X_seq, X_mask_pad, X_mask_activity, X_static, y, ids, num_classes=2, paper_baseline=True):
    if num_classes == 2:
        if paper_baseline:
            # Configuración del Paper: Withdrawn (0) vs Éxito (2,3). Excluye Fail (1).
            valid_mask = (y != 1)
            X_seq = X_seq[valid_mask]
            X_mask_pad = X_mask_pad[valid_mask]
            X_mask_activity = X_mask_activity[valid_mask]
            y, ids = y[valid_mask], ids[valid_mask]
            if X_static is not None:
                X_static = X_static[valid_mask]
            y_formatted = np.where(y == 0, 1, 0)
            logger.info("Configuración 2 clases (Baseline Paper): [0: Pass/Dist] vs [1: Withdrawn]. (Fail eliminado)")
        else:
            # Configuración original: Fail (1) vs Éxito (2,3). Excluye Withdrawn (0).
            valid_mask = (y != 0)
            X_seq = X_seq[valid_mask]
            X_mask_pad = X_mask_pad[valid_mask]
            X_mask_activity = X_mask_activity[valid_mask]
            y, ids = y[valid_mask], ids[valid_mask]
            if X_static is not None:
                X_static = X_static[valid_mask]
            y_formatted = np.where(y == 1, 1, 0)
            logger.info("Configuración 2 clases (Original): [0: Pass/Dist] vs [1: Fail]. (Withdrawn eliminado)")

    elif num_classes == 3:
        # Modo Trinario: Fail (0) vs Withdrawn (1) vs Éxito (2)
        y_formatted = np.where(y >= 2, 2, y)
        logger.info("Configuración 3 clases: [0: Fail], [1: Withdrawn], [2: Pass/Dist]")

    elif num_classes == 4:
        # Modo Cuaternario: Todas las originales
        y_formatted = y
        logger.info("Configuración 4 clases: [0: Fail], [1: Withdrawn], [2: Pass], [3: Distinction]")
    
    else:
        raise ValueError("num_classes debe ser 2, 3 o 4.")
    
    return X_seq, X_mask_pad, X_mask_activity, X_static, y_formatted, ids


def load_and_prepare_split(base_npz: Path, split: str, w_key: int, num_classes: int, paper_baseline: bool, with_static: bool):
    file_path = base_npz / split / f"transformer_uptoW{w_key}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"No se encuentra {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    X_seq = data["X_seq"]
    mask_pad = data["mask_pad"] if "mask_pad" in data.files else data["mask"]
    mask_activity = data["mask_activity"] if "mask_activity" in data.files else mask_pad
    y = data["y"]
    ids = data["ids"]
    X_static = data["X_static"] if with_static else None
    
    X_seq, mask_pad, mask_activity, X_static, y, ids = filter_classes(
        X_seq,
        mask_pad,
        mask_activity,
        X_static,
        y,
        ids,
        num_classes,
        paper_baseline,
    )
    
    return X_seq, mask_pad, mask_activity, X_static, y


@app.command()
def train(
    upto_week: int = typer.Option(5, help="Semana hasta la que utilizar datos (W)"),
    num_classes: int = typer.Option(TRANSFORMER_PARAMS.num_classes, help="Número de clases objetivo (2, 3, o 4)"),
    paper_baseline: bool = typer.Option(True, help="Si num_classes=2, usar config de Paper (1 vs 2+3). False=Original (0 vs 2+3)"),
    batch_size: int = typer.Option(TRANSFORMER_PARAMS.batch_size),
    with_static: bool = typer.Option(TRANSFORMER_PARAMS.with_static, help="Usar variables estáticas"),
    eval_test: bool = typer.Option(False, help="Evaluar también en test (úsalo solo al final)"),
    history_filename: Optional[str] = typer.Option(None, help="Nombre custom para el archivo historial (ej. random_search.json)"),
    latent_d: Optional[int] = typer.Option(None, help="Override latent_d"),
    num_heads: Optional[int] = typer.Option(None, help="Override num_heads"),
    ff_dim: Optional[int] = typer.Option(None, help="Override ff_dim"),
    dropout: Optional[float] = typer.Option(None, help="Override dropout"),
    num_layers: Optional[int] = typer.Option(None, help="Override num_layers"),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning_rate"),
    focal_gamma: Optional[float] = typer.Option(None, help="Override focal_gamma"),
    static_hidden_dim: Optional[int] = typer.Option(None, help="Override static_hidden starting dimension"),
    head_hidden_dim: Optional[int] = typer.Option(None, help="Override head_hidden starting dimension"),
    reduce_lr_patience: Optional[int] = typer.Option(None, help="Override reduce_lr_patience"),
    early_stopping_patience: Optional[int] = typer.Option(None, help="Override early_stopping_patience"),
    seed: Optional[int] = typer.Option(None, help="Random seed para reproducibilidad (numpy/tensorflow)"),
    fast_search: bool = typer.Option(False, help="Disable plots and model saving for fast search")
):
    """
    Entrena el modelo Transformer y evalúa el rendimiento.
    """
    # Usar copia local de hyperparams en lugar de transmutar la base
    hp = copy.deepcopy(TRANSFORMER_PARAMS)
    if seed is not None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        logger.info(f"🌱 Seed fijada a {seed}")

    if latent_d is not None: hp.latent_d = latent_d
    if num_heads is not None: hp.num_heads = num_heads
    if ff_dim is not None: hp.ff_dim = ff_dim
    if dropout is not None: hp.dropout = dropout
    if num_layers is not None: hp.num_layers = num_layers
    if learning_rate is not None: hp.learning_rate = learning_rate
    if focal_gamma is not None: hp.focal_gamma = focal_gamma
    if static_hidden_dim is not None: hp.static_hidden = [static_hidden_dim, static_hidden_dim // 2]
    if head_hidden_dim is not None: hp.head_hidden = [head_hidden_dim, head_hidden_dim // 2]
    if reduce_lr_patience is not None: hp.reduce_lr_patience = reduce_lr_patience
    if early_stopping_patience is not None: hp.early_stopping_patience = early_stopping_patience
    hp.batch_size = batch_size
    base_npz = Path("/workspace/TFM_education_ai_analytics/data/6_transformer_features")
    save_dir = Path(f"/workspace/TFM_education_ai_analytics/reports/transformer_training/week_{upto_week}")
    
    logger.info(f"Cargando datos pre-normalizados desde: {base_npz}")
    
    X_train_seq, train_mask_pad, train_mask_activity, X_train_stat, y_train = load_and_prepare_split(base_npz, "training", upto_week, num_classes, paper_baseline, with_static)
    X_val_seq, val_mask_pad, val_mask_activity, X_val_stat, y_val = load_and_prepare_split(base_npz, "validation", upto_week, num_classes, paper_baseline, with_static)
    if eval_test:
        X_test_seq, test_mask_pad, test_mask_activity, X_test_stat, y_test = load_and_prepare_split(base_npz, "test", upto_week, num_classes, paper_baseline, with_static)
    
    logger.info(f"Train set pre-normalizado -> Seq: {X_train_seq.shape}, Static: {X_train_stat.shape if X_train_stat is not None else 'N/A'}, Y: {y_train.shape}")
    logger.info(f"Val set pre-normalizado   -> Seq: {X_val_seq.shape}, Static: {X_val_stat.shape if X_val_stat is not None else 'N/A'}, Y: {y_val.shape}")
    if eval_test:
        logger.info(f"Test set pre-normalizado  -> Seq: {X_test_seq.shape}, Static: {X_test_stat.shape if X_test_stat is not None else 'N/A'}, Y: {y_test.shape}")
    
    logger.info("\n📊 BALANCEO DE CLASES (Training)")
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    for c, n in zip(train_classes, train_counts):
        logger.info(f"  Clase {c}: {n} ({n/len(y_train)*100:.2f}%)")
    
    # --- Focal Loss ---
    focal_gamma_val = hp.focal_gamma
    if num_classes == 2:
        focal_alpha_val = hp.focal_alpha
    else:
        weights_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        focal_alpha_val = (weights_values / weights_values.sum()).tolist()
    
    def sparse_focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_one_hot = tf.one_hot(y_true, depth=num_classes)
        p_t = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, focal_gamma_val)
        alpha_t = tf.reduce_sum(y_one_hot * tf.constant(focal_alpha_val, dtype=tf.float32), axis=-1)
        ce = -tf.math.log(p_t)
        return tf.reduce_mean(alpha_t * focal_weight * ce)
    
    logger.info(f"🎯 Focal Loss: gamma={focal_gamma_val}, alpha={focal_alpha_val}")
        
    final_training_set = [
        X_train_seq.astype(np.float32),
        train_mask_pad.astype(np.int32),
        train_mask_activity.astype(np.int32),
    ]
    final_validation_set = [
        X_val_seq.astype(np.float32),
        val_mask_pad.astype(np.int32),
        val_mask_activity.astype(np.int32),
    ]
    if with_static:
        final_training_set.append(X_train_stat.astype(np.float32))
        final_validation_set.append(X_val_stat.astype(np.float32))
        
    logger.info("Construyendo modelo...")
    model = GLUTransformerClassifier(
        latent_d=hp.latent_d,
        num_heads=hp.num_heads,
        ff_dim=hp.ff_dim,
        dropout=hp.dropout,
        num_classes=num_classes,
        num_layers=hp.num_layers,
        max_len=X_train_seq.shape[1],
        with_static_features=with_static,
        static_hidden=hp.static_hidden if with_static else [],
        head_hidden=hp.head_hidden
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.learning_rate, 
        clipnorm=hp.clipnorm
    )
    
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if num_classes == 4:
        metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_acc"))
        
    model.compile(optimizer=optimizer, loss=sparse_focal_loss, metrics=metrics)
    
    # --- Custom Callback: Reduce LR on val_loss + Restore Best Weights by val_loss ---
    class ReduceLRWithRestore(tf.keras.callbacks.Callback):
        def __init__(self, lr_monitor="val_loss", restore_monitor="val_loss",
                     patience=5, factor=0.2, min_lr=1e-6, min_delta=1e-4, cooldown=1, verbose=1):
            super().__init__()
            self.lr_monitor = lr_monitor
            self.restore_monitor = restore_monitor
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.min_delta = min_delta
            self.cooldown = cooldown
            self.verbose = verbose

            # Para LR Scheduler (val_loss -> menor es mejor)
            self.best_lr_val = float("inf")
            self.wait = 0
            self.cooldown_counter = 0
            self.reductions = 0

            # Para Checkpoint de Pesos (val_loss -> menor es mejor)
            self.best_restore_val = float("inf")
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            lr_current = logs.get(self.lr_monitor)
            restore_current = logs.get(self.restore_monitor)

            if lr_current is None or restore_current is None:
                return

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            # 1. Guardar mejores pesos (según restore_monitor, menor es mejor)
            if restore_current < self.best_restore_val:
                self.best_restore_val = restore_current
                self.best_weights = self.model.get_weights()

            # 2. Scheduler de LR (según val_loss)
            if lr_current < (self.best_lr_val - self.min_delta):
                self.best_lr_val = lr_current
                self.wait = 0
                return

            if self.cooldown_counter > 0:
                return

            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                if old_lr <= self.min_lr + 1e-12:
                    self.wait = 0
                    return

                new_lr = max(old_lr * self.factor, self.min_lr)

                # Al reducir LR, rollback a los mejores pesos paramétricos
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

                self.model.optimizer.learning_rate.assign(new_lr)
                self.reductions += 1
                self.cooldown_counter = self.cooldown
                self.wait = 0

                if self.verbose:
                    logger.info(
                        f"⚡ Epoch {epoch+1}: rollback(best {self.restore_monitor}={self.best_restore_val:.4f}) "
                        f"+ LR {old_lr:.2e} → {new_lr:.2e} (#{self.reductions})"
                    )

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                if self.verbose:
                    logger.info(f"🏁 Entrenamiento completado. Restaurados pesos con best {self.restore_monitor}={self.best_restore_val:.4f}")
    
    callbacks = [
        ReduceLRWithRestore(
            lr_monitor="val_loss",
            restore_monitor="val_loss",
            patience=hp.reduce_lr_patience,
            factor=hp.reduce_lr_factor,
            min_lr=hp.reduce_lr_min_lr,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=hp.early_stopping_patience, 
            restore_best_weights=False,  # Lo hace la clase ReduceLRWithRestore
            verbose=1
        )
    ]
    
    logger.info("Comenzando entrenamiento...")
    history = model.fit(
        x=final_training_set,
        y=y_train,
        validation_data=(final_validation_set, y_val),
        epochs=hp.epochs,
        batch_size=batch_size,
        # Focal loss maneja el desbalanceo internamente
        callbacks=callbacks,
        verbose=2
    )
    
    logger.info("Evaluando modelo en Validation Set...")
    results = model.evaluate(final_validation_set, y_val, verbose=0)
    logger.info(f"Loss final Val: {results[0]:.4f} | Accuracy: {results[1]:.4f}")
    
    y_probs = model.predict(final_validation_set, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    
    logger.info("\n[Balanced Accuracy VAL]: " + str(balanced_accuracy_score(y_val, y_pred)))
    logger.info("\n[Confusion Matrix VAL]\n" + str(confusion_matrix(y_val, y_pred)))
    logger.info("\n[Classification Report VAL]\n" + str(classification_report(y_val, y_pred, digits=4)))
    
    try:
        if num_classes == 2:
            auc_val = roc_auc_score(y_val, y_probs[:, 1])
        else:
            auc_val = roc_auc_score(y_val, y_probs, multi_class='ovr')
        logger.info(f"AUC VAL (OVR): {auc_val:.4f}")
    except ValueError as e:
        logger.warning(f"No se pudo calcular AUC en VAL: {e}")
        
    if eval_test:
        logger.info("--------- Evaluando modelo en Test Set (FINAL) ---------")
        final_test_set = [
            X_test_seq.astype(np.float32),
            test_mask_pad.astype(np.int32),
            test_mask_activity.astype(np.int32),
        ]
        if with_static:
            final_test_set.append(X_test_stat.astype(np.float32))
            
        results_test = model.evaluate(final_test_set, y_test, verbose=0)
        logger.info(f"Loss final TEST: {results_test[0]:.4f} | Accuracy TEST: {results_test[1]:.4f}")
        
        y_probs_test = model.predict(final_test_set, verbose=0)
        y_pred_test = np.argmax(y_probs_test, axis=1)
        
        logger.info("\n[Balanced Accuracy TEST]: " + str(balanced_accuracy_score(y_test, y_pred_test)))
        logger.info("\n[Confusion Matrix TEST]\n" + str(confusion_matrix(y_test, y_pred_test)))
        logger.info("\n[Classification Report TEST]\n" + str(classification_report(y_test, y_pred_test, digits=4)))
        
        try:
            if num_classes == 2:
                auc_test = roc_auc_score(y_test, y_probs_test[:, 1])
            else:
                auc_test = roc_auc_score(y_test, y_probs_test, multi_class='ovr')
            logger.info(f"AUC TEST (OVR): {auc_test:.4f}")
        except ValueError as e:
            logger.warning(f"No se pudo calcular AUC en TEST: {e}")
        
    if not fast_search:
        # Plotting
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('dark_background')
        plt.rcParams.update({"figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e"})
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Progreso de Entrenamiento (Ventana: {upto_week} | Clases: {num_classes})", fontsize=16, fontweight='bold', y=1.05)
        
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Evolución de la Función de Pérdida', pad=10)
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss (Sparse Categorical CE)')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        axes[1].plot(history.history.get('accuracy', []), label='Train Accuracy', linewidth=2)
        axes[1].plot(history.history.get('val_accuracy', []), label='Validation Accuracy', linewidth=2)
        
        if "top_2_acc" in history.history:
            axes[1].plot(history.history["top_2_acc"], label="Train Top-2", linewidth=2, linestyle='--')
        if "val_top_2_acc" in history.history:
            axes[1].plot(history.history["val_top_2_acc"], label="Val Top-2", linewidth=2, linestyle='--')
        axes[1].set_title('Evolución de la Precisión', pad=10)
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.3)
        
        sns.despine(fig)
        plt.tight_layout()
        
        plot_path = save_dir / f"plot_uptoW{upto_week}_{num_classes}clases.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Gráfico guardado en: {plot_path}")
        
        # Guardar modelo entrenado
        model_dir = Path("/workspace/TFM_education_ai_analytics/models/transformers")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"transformer_uptoW{upto_week}_{num_classes}clases.keras"
        model.save(model_path)
        logger.info(f"✅ Modelo final guardado en: {model_path}")
    

    # Guardar métricas y configuración en historial via hyperparams
    if num_classes == 2:
        val_precision = float(precision_score(y_val, y_pred, pos_label=1, average='binary'))
        val_recall = float(recall_score(y_val, y_pred, pos_label=1, average='binary'))
        val_f1 = float(f1_score(y_val, y_pred, pos_label=1, average='binary'))
    else:
        val_precision = float(precision_score(y_val, y_pred, average='macro'))
        val_recall = float(recall_score(y_val, y_pred, average='macro'))
        val_f1 = float(f1_score(y_val, y_pred, average='macro'))

    test_loss = float(results_test[0]) if eval_test else None
    test_acc = float(results_test[1]) if eval_test else None
    test_bacc = balanced_accuracy_score(y_test, y_pred_test) if eval_test else None
    test_auc_val = float(auc_test) if eval_test and 'auc_test' in locals() else None
    
    if eval_test:
        if num_classes == 2:
            test_precision = float(precision_score(y_test, y_pred_test, pos_label=1, average='binary'))
            test_recall = float(recall_score(y_test, y_pred_test, pos_label=1, average='binary'))
            test_f1 = float(f1_score(y_test, y_pred_test, pos_label=1, average='binary'))
        else:
            test_precision = float(precision_score(y_test, y_pred_test, average='macro'))
            test_recall = float(recall_score(y_test, y_pred_test, average='macro'))
            test_f1 = float(f1_score(y_test, y_pred_test, average='macro'))
    else:
        test_precision = test_recall = test_f1 = None

    # Payload explícito para automatización (Random Search, Optuna, etc.)
    val_metrics = {
        "val_loss": float(results[0]),
        "val_accuracy": float(results[1]),
        "val_balanced_acc": float(balanced_accuracy_score(y_val, y_pred)),
        "val_f1": float(val_f1 if 'val_f1' in locals() and val_f1 is not None else 0),
        "val_recall": float(val_recall if 'val_recall' in locals() and val_recall is not None else 0),
        "val_precision": float(val_precision if 'val_precision' in locals() and val_precision is not None else 0),
        "val_auc": float(auc_val) if 'auc_val' in locals() else 0,
    }
    logger.info("VAL_METRICS: " + json.dumps(val_metrics))

    if not fast_search:
        hp.save_experiment(
            save_dir=save_dir,
            upto_week=upto_week,
            paper_baseline=paper_baseline,
            val_loss=results[0],
            val_acc=results[1],
            val_balanced_acc=balanced_accuracy_score(y_val, y_pred),
            val_auc=float(auc_val) if 'auc_val' in locals() else None,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            test_loss=test_loss,
            test_acc=test_acc,
            test_balanced_acc=test_bacc,
            test_auc=test_auc_val,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1=test_f1,
            history_filename=history_filename
        )
        
        history_file_used = history_filename if history_filename else f"experiments_history_{num_classes}clases.json"
        # ------------------
        # Disparar script de comparativa
        # ------------------
        try:
            from compare_experiments import compare_experiments
            logger.info("\n" + "="*80)
            compare_experiments(history_path=save_dir / history_file_used)
        except Exception as e:
            logger.error(f"No se pudo generar la comparativa automática: {e}")

if __name__ == "__main__":
    app()
