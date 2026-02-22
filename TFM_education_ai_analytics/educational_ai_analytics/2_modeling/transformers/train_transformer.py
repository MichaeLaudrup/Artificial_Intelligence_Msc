import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import typer
import numpy as np
import tensorflow as tf
from pathlib import Path
from dataclasses import asdict
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from typing import Optional
import copy

from educational_ai_analytics.config import W_WINDOWS

try:
    from .transformer_GLU_classifier import GLUTransformerClassifier
    from .hyperparams import TRANSFORMER_PARAMS
    from .utils.training_config import build_runtime_config_from_cli
    from .utils.thresholding import select_binary_threshold_with_constraints
    from .utils.training_callbacks import ReduceLRWithRestore, KeepBestValBalancedAcc
except ImportError:
    # Fallback para ejecución directa del archivo (python path/to/train_transformer.py)
    from transformer_GLU_classifier import GLUTransformerClassifier
    from hyperparams import TRANSFORMER_PARAMS
    from utils.training_config import build_runtime_config_from_cli
    from utils.thresholding import select_binary_threshold_with_constraints
    from utils.training_callbacks import ReduceLRWithRestore, KeepBestValBalancedAcc

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
    features_cluster = data["features_cluster"] if "features_cluster" in data.files else np.array([], dtype=object)
    static_feature_names = data["static_feature_names"] if "static_feature_names" in data.files else np.array([], dtype=object)
    static_feature_sources = data["static_feature_sources"] if "static_feature_sources" in data.files else np.array([], dtype=object)
    cluster_feature_dim = int(len(features_cluster)) if with_static and X_static is not None else 0
    
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
    
    return X_seq, mask_pad, mask_activity, X_static, y, cluster_feature_dim, static_feature_names, static_feature_sources




@app.command()
def train(
    upto_week: int = typer.Option(TRANSFORMER_PARAMS.upto_week, help="Semana hasta la que utilizar datos (W)"),
    num_classes: int = typer.Option(TRANSFORMER_PARAMS.num_classes, help="Número de clases objetivo (2, 3, o 4)"),
    paper_baseline: bool = typer.Option(True, help="Si num_classes=2, usar config de Paper (1 vs 2+3). False=Original (0 vs 2+3)"),
    batch_size: int = typer.Option(TRANSFORMER_PARAMS.batch_size),
    with_static: bool = typer.Option(TRANSFORMER_PARAMS.with_static, help="Usar variables estáticas"),
    use_clustering_features: bool = typer.Option(TRANSFORMER_PARAMS.use_clustering_features, help="Usar features de clustering en X_static (todo o nada)"),
    accumulated_uptow: bool = typer.Option(TRANSFORMER_PARAMS.accumulated_uptow, help="Usar features estáticas acumuladas ae_uptow (todo o nada)"),
    eval_test: bool = typer.Option(False, help="Evaluar también en test (úsalo solo al final)"),
    history_filename: Optional[str] = typer.Option(None, help="Nombre custom para el archivo historial (ej. random_search.json)"),
    latent_d: Optional[int] = typer.Option(None, help="Override latent_d"),
    num_heads: Optional[int] = typer.Option(None, help="Override num_heads"),
    ff_dim: Optional[int] = typer.Option(None, help="Override ff_dim"),
    dropout: Optional[float] = typer.Option(None, help="Override dropout"),
    num_layers: Optional[int] = typer.Option(None, help="Override num_layers"),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning_rate"),
    focal_gamma: Optional[float] = typer.Option(None, help="Override focal_gamma"),
    focal_alpha_pos: Optional[float] = typer.Option(None, help="Override alpha clase positiva para focal loss binaria (0-1)"),
    static_hidden_dim: Optional[int] = typer.Option(None, help="Override static_hidden starting dimension"),
    head_hidden_dim: Optional[int] = typer.Option(None, help="Override head_hidden starting dimension"),
    reduce_lr_patience: Optional[int] = typer.Option(None, help="Override reduce_lr_patience"),
    early_stopping_patience: Optional[int] = typer.Option(None, help="Override early_stopping_patience"),
    seed: Optional[int] = typer.Option(None, help="Random seed para reproducibilidad (numpy/tensorflow)"),
    tune_threshold: bool = typer.Option(False, help="Afinar umbral binario en validación con restricciones de accuracy/precision"),
    threshold_acc_min: float = typer.Option(0.80, help="Accuracy mínima para umbral binario"),
    threshold_prec_min: float = typer.Option(0.67, help="Precision mínima para umbral binario"),
    threshold_objective: str = typer.Option("recall", help="Objetivo de tuning de umbral: recall|f1|balanced_accuracy"),
    threshold_min: float = typer.Option(0.20, help="Umbral mínimo a explorar"),
    threshold_max: float = typer.Option(0.80, help="Umbral máximo a explorar"),
    threshold_points: int = typer.Option(301, help="Número de puntos en la rejilla de umbrales"),
    threshold_fallback: float = typer.Option(0.50, help="Umbral por defecto si no hay candidatos factibles"),
    run_compare: bool = typer.Option(True, help="Ejecutar compare_experiments al finalizar"),
    config_json: Optional[Path] = typer.Option(None, help="Ruta a JSON con TrainingConfig para inyección externa"),
    metrics_out: Optional[Path] = typer.Option(None, help="Ruta de salida JSON con métricas de esta corrida"),
    fast_search: bool = typer.Option(False, help="Disable plots and model saving for fast search")
):
    """
    Entrena el modelo Transformer y evalúa el rendimiento.
    """
    runtime_cfg = build_runtime_config_from_cli(locals(), config_json=config_json)
    if config_json is not None:
        logger.info(f"📥 Config inyectada desde JSON: {config_json}")
    cfg = runtime_cfg

    valid_threshold_objectives = {"recall", "f1", "balanced_accuracy"}
    cfg.threshold_objective = str(cfg.threshold_objective).strip().lower()
    if cfg.threshold_objective not in valid_threshold_objectives:
        raise ValueError(
            f"threshold_objective inválido: {cfg.threshold_objective}. "
            f"Usa uno de: {', '.join(sorted(valid_threshold_objectives))}"
        )

    # Usar copia local de hyperparams en lugar de transmutar la base
    hp = copy.deepcopy(TRANSFORMER_PARAMS)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        tf.keras.utils.set_random_seed(cfg.seed)
        logger.info(f"🌱 Seed fijada a {cfg.seed}")

    overrides = {
        "latent_d": cfg.latent_d,
        "num_heads": cfg.num_heads,
        "ff_dim": cfg.ff_dim,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
        "learning_rate": cfg.learning_rate,
        "focal_gamma": cfg.focal_gamma,
        "reduce_lr_patience": cfg.reduce_lr_patience,
        "early_stopping_patience": cfg.early_stopping_patience,
    }
    for attr, value in overrides.items():
        if value is not None:
            setattr(hp, attr, value)

    if cfg.static_hidden_dim is not None:
        hp.static_hidden = [cfg.static_hidden_dim, cfg.static_hidden_dim // 2]
    if cfg.head_hidden_dim is not None:
        hp.head_hidden = [cfg.head_hidden_dim, cfg.head_hidden_dim // 2]
    if cfg.num_classes == 2 and cfg.focal_alpha_pos is not None:
        if not (0.0 < cfg.focal_alpha_pos < 1.0):
            raise ValueError("focal_alpha_pos debe estar en (0, 1)")
        hp.focal_alpha = [1.0 - float(cfg.focal_alpha_pos), float(cfg.focal_alpha_pos)]
    hp.batch_size = cfg.batch_size
    base_npz = Path("/workspace/TFM_education_ai_analytics/data/6_transformer_features")
    save_dir = Path(f"/workspace/TFM_education_ai_analytics/reports/transformer_training/week_{cfg.upto_week}")
    
    logger.info(f"Cargando datos pre-normalizados desde: {base_npz}")
    
    X_train_seq, train_mask_pad, train_mask_activity, X_train_stat, y_train, train_cluster_dim, train_static_names, train_static_sources = load_and_prepare_split(base_npz, "training", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static)
    X_val_seq, val_mask_pad, val_mask_activity, X_val_stat, y_val, val_cluster_dim, val_static_names, val_static_sources = load_and_prepare_split(base_npz, "validation", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static)
    if cfg.eval_test:
        X_test_seq, test_mask_pad, test_mask_activity, X_test_stat, y_test, test_cluster_dim, test_static_names, test_static_sources = load_and_prepare_split(base_npz, "test", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static)

    use_static_in_model = cfg.with_static

    def _is_accumulated_name(col: str) -> bool:
        col = str(col)
        accumulated_prefixes = (
            "clicks_",
            "weeks_since_",
            "streak_",
            "total_weighted_",
            "last_week_",
            "momentum",
            "regularity",
            "weekend_share",
            "distinct_activity",
            "recency_",
            "engagement_",
            "pass_ratio",
            "late_ratio",
            "submission_",
            "avg_score",
            "active_weeks",
        )
        return col.startswith(accumulated_prefixes)

    def _is_cluster_name(col: str) -> bool:
        col = str(col)
        return col.startswith("p_cluster_") or col == "entropy_norm"

    def _cluster_drop_mask(feature_names: np.ndarray, feature_sources: np.ndarray, n_features: int, cluster_dim: int) -> np.ndarray:
        if n_features == 0:
            return np.zeros((0,), dtype=bool)

        if feature_sources.size == n_features:
            src = np.array(feature_sources).astype(str)
            return src == "cluster"

        if feature_names.size == n_features:
            names = np.array(feature_names).astype(str)
            return np.array([_is_cluster_name(c) for c in names], dtype=bool)

        # Fallback para npz legacy sin metadata: asume bloque cluster al inicio
        fallback = np.zeros((n_features,), dtype=bool)
        if cluster_dim > 0:
            fallback[:min(cluster_dim, n_features)] = True
        return fallback

    def _filter_static_block(
        x_static: Optional[np.ndarray],
        cluster_dim: int,
        feature_names: np.ndarray,
        feature_sources: np.ndarray,
        keep_clusters: bool,
        keep_accumulated: bool,
        split_name: str,
    ) -> Optional[np.ndarray]:
        if x_static is None:
            return None

        n_features = x_static.shape[1]
        keep_mask = np.ones(n_features, dtype=bool)
        drop_cluster_mask = _cluster_drop_mask(feature_names, feature_sources, n_features, cluster_dim)
        n_cluster_cols = int(drop_cluster_mask.sum())

        if not keep_clusters:
            keep_mask &= ~drop_cluster_mask
            if n_cluster_cols == 0:
                logger.warning(
                    f"⚠️ [{split_name}] use_clustering_features=False pero no se detectaron columnas cluster en X_static."
                )

        if not keep_accumulated:
            if feature_sources.size == n_features:
                src = np.array(feature_sources).astype(str)
                keep_mask &= (src != "accumulated_uptow")
            elif feature_names.size == n_features:
                names = np.array(feature_names).astype(str)
                drop_acc = np.array([_is_accumulated_name(c) for c in names], dtype=bool)
                keep_mask &= ~drop_acc
            else:
                logger.warning("⚠️ No hay metadata de columnas estáticas; no se puede filtrar accumulated_uptow de forma segura")

        logger.info(
            f"[{split_name}] Static ablation -> total={n_features} | cluster_detected={n_cluster_cols} | "
            f"kept={int(keep_mask.sum())} | removed={int((~keep_mask).sum())}"
        )

        if keep_mask.sum() == 0:
            return np.zeros((x_static.shape[0], 0), dtype=x_static.dtype)
        return x_static[:, keep_mask]

    if use_static_in_model and (not cfg.use_clustering_features or not cfg.accumulated_uptow):
        if not cfg.use_clustering_features:
            logger.info("🧪 Ablation: clustering features DESACTIVADAS (todo o nada)")
        if not cfg.accumulated_uptow:
            logger.info("🧪 Ablation: accumulated_uptow DESACTIVADAS (todo o nada)")

        X_train_stat = _filter_static_block(
            X_train_stat,
            train_cluster_dim,
            train_static_names,
            train_static_sources,
            keep_clusters=cfg.use_clustering_features,
            keep_accumulated=cfg.accumulated_uptow,
            split_name="training",
        )
        X_val_stat = _filter_static_block(
            X_val_stat,
            val_cluster_dim,
            val_static_names,
            val_static_sources,
            keep_clusters=cfg.use_clustering_features,
            keep_accumulated=cfg.accumulated_uptow,
            split_name="validation",
        )
        if cfg.eval_test:
            X_test_stat = _filter_static_block(
                X_test_stat,
                test_cluster_dim,
                test_static_names,
                test_static_sources,
                keep_clusters=cfg.use_clustering_features,
                keep_accumulated=cfg.accumulated_uptow,
                split_name="test",
            )

        if X_train_stat is None or X_train_stat.shape[1] == 0:
            logger.warning("⚠️ No quedan variables estáticas tras aplicar filtros; se desactiva with_static para esta corrida")
            use_static_in_model = False
            X_train_stat = None
            X_val_stat = None
            if cfg.eval_test:
                X_test_stat = None
    
    logger.info(f"Train set pre-normalizado -> Seq: {X_train_seq.shape}, Static: {X_train_stat.shape if X_train_stat is not None else 'N/A'}, Y: {y_train.shape}")
    logger.info(f"Val set pre-normalizado   -> Seq: {X_val_seq.shape}, Static: {X_val_stat.shape if X_val_stat is not None else 'N/A'}, Y: {y_val.shape}")
    if cfg.eval_test:
        logger.info(f"Test set pre-normalizado  -> Seq: {X_test_seq.shape}, Static: {X_test_stat.shape if X_test_stat is not None else 'N/A'}, Y: {y_test.shape}")
    
    logger.info("\n📊 BALANCEO DE CLASES (Training)")
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    for c, n in zip(train_classes, train_counts):
        logger.info(f"  Clase {c}: {n} ({n/len(y_train)*100:.2f}%)")
    
    # --- Focal Loss ---
    focal_gamma_val = hp.focal_gamma
    if cfg.num_classes == 2:
        focal_alpha_val = hp.focal_alpha
    else:
        weights_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        focal_alpha_val = (weights_values / weights_values.sum()).tolist()
    
    def sparse_focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_one_hot = tf.one_hot(y_true, depth=cfg.num_classes)
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
    if use_static_in_model:
        final_training_set.append(X_train_stat.astype(np.float32))
        final_validation_set.append(X_val_stat.astype(np.float32))
        
    logger.info("Construyendo modelo...")
    model = GLUTransformerClassifier(
        latent_d=hp.latent_d,
        num_heads=hp.num_heads,
        ff_dim=hp.ff_dim,
        dropout=hp.dropout,
        num_classes=cfg.num_classes,
        num_layers=hp.num_layers,
        max_len=X_train_seq.shape[1],
        with_static_features=use_static_in_model,
        static_hidden=hp.static_hidden if use_static_in_model else [],
        head_hidden=hp.head_hidden
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.learning_rate, 
        clipnorm=hp.clipnorm
    )
    
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if cfg.num_classes == 4:
        metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_acc"))
        
    model.compile(optimizer=optimizer, loss=sparse_focal_loss, metrics=metrics)
    
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
        ),
        KeepBestValBalancedAcc(
            val_inputs=final_validation_set,
            val_targets=y_val,
            restore_on_train_end=True,
            verbose=1,
        ),
    ]
    
    logger.info("Comenzando entrenamiento...")
    history = model.fit(
        x=final_training_set,
        y=y_train,
        validation_data=(final_validation_set, y_val),
        epochs=hp.epochs,
        batch_size=cfg.batch_size,
        # Focal loss maneja el desbalanceo internamente
        callbacks=callbacks,
        verbose=2
    )
    
    logger.info("Evaluando modelo en Validation Set...")
    results = model.evaluate(final_validation_set, y_val, verbose=0)
    logger.info(f"Loss final Val: {results[0]:.4f} | Accuracy: {results[1]:.4f}")
    
    y_probs = model.predict(final_validation_set, verbose=0)
    selected_threshold = None
    if cfg.num_classes == 2 and cfg.tune_threshold:
        p_pos = y_probs[:, 1]
        best_thr, candidates = select_binary_threshold_with_constraints(
            y_true=y_val,
            p_pos=p_pos,
            acc_min=cfg.threshold_acc_min,
            prec_min=cfg.threshold_prec_min,
            objective=cfg.threshold_objective,
            t_min=cfg.threshold_min,
            t_max=cfg.threshold_max,
            n_points=cfg.threshold_points,
        )
        if best_thr is None:
            selected_threshold = float(cfg.threshold_fallback)
            logger.warning(
                f"⚠️ Threshold tuning: sin candidatos con acc>={cfg.threshold_acc_min:.3f} y prec>={cfg.threshold_prec_min:.3f}. "
                f"Usando fallback={selected_threshold:.3f}"
            )
        else:
            selected_threshold = float(best_thr["threshold"])
            logger.info(
                f"🎚️ Threshold tuning: {len(candidates)} candidatos factibles | "
                f"obj={cfg.threshold_objective} | best={selected_threshold:.3f} | acc={best_thr['accuracy']:.4f} | "
                f"prec={best_thr['precision']:.4f} | rec={best_thr['recall']:.4f}"
            )
        y_pred = (p_pos >= selected_threshold).astype(int)
    else:
        y_pred = np.argmax(y_probs, axis=1)
    
    logger.info("\n[Balanced Accuracy VAL]: " + str(balanced_accuracy_score(y_val, y_pred)))
    logger.info("\n[Confusion Matrix VAL]\n" + str(confusion_matrix(y_val, y_pred)))
    logger.info("\n[Classification Report VAL]\n" + str(classification_report(y_val, y_pred, digits=4)))
    
    try:
        if cfg.num_classes == 2:
            auc_val = roc_auc_score(y_val, y_probs[:, 1])
        else:
            auc_val = roc_auc_score(y_val, y_probs, multi_class='ovr')
        logger.info(f"AUC VAL (OVR): {auc_val:.4f}")
    except ValueError as e:
        logger.warning(f"No se pudo calcular AUC en VAL: {e}")
        
    if cfg.eval_test:
        logger.info("--------- Evaluando modelo en Test Set (FINAL) ---------")
        final_test_set = [
            X_test_seq.astype(np.float32),
            test_mask_pad.astype(np.int32),
            test_mask_activity.astype(np.int32),
        ]
        if use_static_in_model:
            final_test_set.append(X_test_stat.astype(np.float32))
            
        results_test = model.evaluate(final_test_set, y_test, verbose=0)
        logger.info(f"Loss final TEST: {results_test[0]:.4f} | Accuracy TEST: {results_test[1]:.4f}")
        
        y_probs_test = model.predict(final_test_set, verbose=0)
        if cfg.num_classes == 2 and selected_threshold is not None:
            y_pred_test = (y_probs_test[:, 1] >= selected_threshold).astype(int)
            logger.info(f"🎯 Aplicando en TEST el threshold aprendido en VAL: {selected_threshold:.3f}")
        else:
            y_pred_test = np.argmax(y_probs_test, axis=1)
        
        logger.info("\n[Balanced Accuracy TEST]: " + str(balanced_accuracy_score(y_test, y_pred_test)))
        logger.info("\n[Confusion Matrix TEST]\n" + str(confusion_matrix(y_test, y_pred_test)))
        logger.info("\n[Classification Report TEST]\n" + str(classification_report(y_test, y_pred_test, digits=4)))
        
        try:
            if cfg.num_classes == 2:
                auc_test = roc_auc_score(y_test, y_probs_test[:, 1])
            else:
                auc_test = roc_auc_score(y_test, y_probs_test, multi_class='ovr')
            logger.info(f"AUC TEST (OVR): {auc_test:.4f}")
        except ValueError as e:
            logger.warning(f"No se pudo calcular AUC en TEST: {e}")
        
    if not cfg.fast_search:
        # Plotting
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('dark_background')
        plt.rcParams.update({"figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e"})
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Progreso de Entrenamiento (Ventana: {cfg.upto_week} | Clases: {cfg.num_classes})", fontsize=16, fontweight='bold', y=1.05)
        
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
        
        plot_path = save_dir / f"plot_uptoW{cfg.upto_week}_{cfg.num_classes}clases.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Gráfico guardado en: {plot_path}")
        
        # Guardar modelo entrenado
        model_dir = Path("/workspace/TFM_education_ai_analytics/models/transformers")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"transformer_uptoW{cfg.upto_week}_{cfg.num_classes}clases.keras"
        model.save(model_path)
        logger.info(f"✅ Modelo final guardado en: {model_path}")
    

    # Guardar métricas y configuración en historial via hyperparams
    if cfg.num_classes == 2:
        val_precision = float(precision_score(y_val, y_pred, pos_label=1, average='binary'))
        val_recall = float(recall_score(y_val, y_pred, pos_label=1, average='binary'))
        val_f1 = float(f1_score(y_val, y_pred, pos_label=1, average='binary'))
    else:
        val_precision = float(precision_score(y_val, y_pred, average='macro'))
        val_recall = float(recall_score(y_val, y_pred, average='macro'))
        val_f1 = float(f1_score(y_val, y_pred, average='macro'))

    test_loss = float(results_test[0]) if cfg.eval_test else None
    test_acc = float(results_test[1]) if cfg.eval_test else None
    test_bacc = balanced_accuracy_score(y_test, y_pred_test) if cfg.eval_test else None
    test_auc_val = float(auc_test) if cfg.eval_test and 'auc_test' in locals() else None
    
    if cfg.eval_test:
        if cfg.num_classes == 2:
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
        "val_threshold": float(selected_threshold) if selected_threshold is not None else None,
    }
    logger.info("VAL_METRICS: " + json.dumps(val_metrics))

    if not cfg.fast_search:
        hp.save_experiment(
            save_dir=save_dir,
            upto_week=cfg.upto_week,
            paper_baseline=cfg.paper_baseline,
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
            history_filename=cfg.history_filename
        )
        
        history_file_used = cfg.history_filename if cfg.history_filename else f"experiments_history_{cfg.num_classes}clases.json"
        # ------------------
        # Disparar script de comparativa
        # ------------------
        if cfg.run_compare:
            try:
                try:
                    from .utils.compare_experiments import compare_experiments
                except ImportError:
                    from utils.compare_experiments import compare_experiments
                logger.info("\n" + "="*80)
                compare_experiments(history_path=save_dir / history_file_used)
            except Exception as e:
                logger.error(f"No se pudo generar la comparativa automática: {e}")

    if metrics_out is not None:
        payload = {
            "config": asdict(runtime_cfg),
            "val_metrics": val_metrics,
            "test_metrics": {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_balanced_acc": float(test_bacc) if test_bacc is not None else None,
                "test_auc": test_auc_val,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            },
            "selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"💾 Metrics JSON guardado en: {metrics_out}")

if __name__ == "__main__":
    app()
