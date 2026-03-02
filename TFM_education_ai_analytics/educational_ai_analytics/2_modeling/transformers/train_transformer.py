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

def _normalize_binary_mode(paper_baseline: bool, binary_mode: Optional[str]) -> str:
    if binary_mode is None:
        return "paper" if paper_baseline else "original"
    mode = str(binary_mode).strip().lower()
    aliases = {
        "paper": "paper",
        "baseline": "paper",
        "original": "original",
        "success_vs_risk": "success_vs_risk",
        "risk": "success_vs_risk",
        "passdist_vs_failwithdraw": "success_vs_risk",
    }
    if mode not in aliases:
        raise ValueError(
            f"binary_mode inválido: {binary_mode}. Usa uno de: paper|baseline|original|success_vs_risk"
        )
    return aliases[mode]


def filter_classes(
    X_seq,
    X_mask_pad,
    X_mask_activity,
    X_static,
    y,
    ids,
    num_classes=2,
    paper_baseline=True,
    binary_mode: Optional[str] = None,
):
    if num_classes == 2:
        mode = _normalize_binary_mode(paper_baseline=paper_baseline, binary_mode=binary_mode)
        if mode == "paper":
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
        elif mode == "original":
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
        else:
            # Caso especial binario: Éxito (2,3) vs Riesgo (0,1). No elimina clases.
            y_formatted = np.where(y >= 2, 0, 1)
            logger.info("Configuración 2 clases (Success vs Risk): [0: Pass/Dist] vs [1: Fail/Withdrawn].")

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


def load_and_prepare_split(
    base_npz: Path,
    split: str,
    w_key: int,
    num_classes: int,
    paper_baseline: bool,
    with_static: bool,
    binary_mode: Optional[str],
):
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
        binary_mode,
    )
    
    return X_seq, mask_pad, mask_activity, X_static, y, cluster_feature_dim, static_feature_names, static_feature_sources




@app.command()
def train(
    upto_week: int = typer.Option(TRANSFORMER_PARAMS.upto_week, help="Semana hasta la que utilizar datos (W)"),
    num_classes: int = typer.Option(TRANSFORMER_PARAMS.num_classes, help="Número de clases objetivo (2, 3, o 4)"),
    paper_baseline: bool = typer.Option(True, help="Si num_classes=2, usar config de Paper (1 vs 2+3). False=Original (0 vs 2+3)"),
    binary_mode: Optional[str] = typer.Option(
        TRANSFORMER_PARAMS.binary_mode,
        help="Modo binario cuando num_classes=2: paper|original|success_vs_risk (pass/dist vs fail/withdraw)",
    ),
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
    selected_binary_mode = _normalize_binary_mode(cfg.paper_baseline, cfg.binary_mode) if cfg.num_classes == 2 else None
    target_tag = f"{cfg.num_classes}clases_{selected_binary_mode}" if cfg.num_classes == 2 else f"{cfg.num_classes}clases"

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
    
    X_train_seq, train_mask_pad, train_mask_activity, X_train_stat, y_train, train_cluster_dim, train_static_names, train_static_sources = load_and_prepare_split(base_npz, "training", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static, cfg.binary_mode)
    X_val_seq, val_mask_pad, val_mask_activity, X_val_stat, y_val, val_cluster_dim, val_static_names, val_static_sources = load_and_prepare_split(base_npz, "validation", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static, cfg.binary_mode)
    if cfg.eval_test:
        X_test_seq, test_mask_pad, test_mask_activity, X_test_stat, y_test, test_cluster_dim, test_static_names, test_static_sources = load_and_prepare_split(base_npz, "test", cfg.upto_week, cfg.num_classes, cfg.paper_baseline, cfg.with_static, cfg.binary_mode)

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
    alpha_candidate = [float(a) for a in hp.focal_alpha] if hp.focal_alpha is not None else []
    if len(alpha_candidate) != int(cfg.num_classes):
        raise ValueError(
            "Configuración inválida: hp.focal_alpha debe tener exactamente num_classes valores "
            f"(num_classes={cfg.num_classes}, len={len(alpha_candidate)}, valor={hp.focal_alpha})."
        )
    focal_alpha_val = alpha_candidate
    
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

    val_bal_acc = float(balanced_accuracy_score(y_val, y_pred))
    val_top2_acc = None
    if cfg.num_classes == 4:
        top2_idx = np.argsort(y_probs, axis=1)[:, -2:]
        val_top2_acc = float(np.mean(np.any(top2_idx == y_val.reshape(-1, 1), axis=1)))
    
    logger.info("\n[Balanced Accuracy VAL]: " + str(val_bal_acc))
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

    if cfg.num_classes == 2:
        val_precision = float(precision_score(y_val, y_pred, pos_label=1, average='binary'))
        val_recall = float(recall_score(y_val, y_pred, pos_label=1, average='binary'))
        val_f1 = float(f1_score(y_val, y_pred, pos_label=1, average='binary'))
        val_precision_macro = val_recall_macro = val_f1_macro = None
        val_precision_weighted = val_recall_weighted = val_f1_weighted = None
    else:
        val_precision_macro = float(precision_score(y_val, y_pred, average='macro'))
        val_recall_macro = float(recall_score(y_val, y_pred, average='macro'))
        val_f1_macro = float(f1_score(y_val, y_pred, average='macro'))
        val_precision_weighted = float(precision_score(y_val, y_pred, average='weighted'))
        val_recall_weighted = float(recall_score(y_val, y_pred, average='weighted'))
        val_f1_weighted = float(f1_score(y_val, y_pred, average='weighted'))
        val_precision = val_precision_macro
        val_recall = val_recall_macro
        val_f1 = val_f1_macro
        logger.info(
            f"[VAL Macro] precision={val_precision_macro:.4f} | recall={val_recall_macro:.4f} | f1={val_f1_macro:.4f}"
        )
        logger.info(
            f"[VAL Weighted] precision={val_precision_weighted:.4f} | recall={val_recall_weighted:.4f} | f1={val_f1_weighted:.4f}"
        )
        if val_top2_acc is not None:
            logger.info(f"[VAL Top-2 Accuracy] {val_top2_acc:.4f}")
        
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

        test_bal_acc_val = float(balanced_accuracy_score(y_test, y_pred_test))
        test_top2_acc = None
        if cfg.num_classes == 4:
            top2_idx_test = np.argsort(y_probs_test, axis=1)[:, -2:]
            test_top2_acc = float(np.mean(np.any(top2_idx_test == y_test.reshape(-1, 1), axis=1)))
        
        logger.info("\n[Balanced Accuracy TEST]: " + str(test_bal_acc_val))
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

        if cfg.num_classes == 2:
            test_precision = float(precision_score(y_test, y_pred_test, pos_label=1, average='binary'))
            test_recall = float(recall_score(y_test, y_pred_test, pos_label=1, average='binary'))
            test_f1 = float(f1_score(y_test, y_pred_test, pos_label=1, average='binary'))
            test_precision_macro = test_recall_macro = test_f1_macro = None
            test_precision_weighted = test_recall_weighted = test_f1_weighted = None
        else:
            test_precision_macro = float(precision_score(y_test, y_pred_test, average='macro'))
            test_recall_macro = float(recall_score(y_test, y_pred_test, average='macro'))
            test_f1_macro = float(f1_score(y_test, y_pred_test, average='macro'))
            test_precision_weighted = float(precision_score(y_test, y_pred_test, average='weighted'))
            test_recall_weighted = float(recall_score(y_test, y_pred_test, average='weighted'))
            test_f1_weighted = float(f1_score(y_test, y_pred_test, average='weighted'))
            test_precision = test_precision_macro
            test_recall = test_recall_macro
            test_f1 = test_f1_macro
            logger.info(
                f"[TEST Macro] precision={test_precision_macro:.4f} | recall={test_recall_macro:.4f} | f1={test_f1_macro:.4f}"
            )
            logger.info(
                f"[TEST Weighted] precision={test_precision_weighted:.4f} | recall={test_recall_weighted:.4f} | f1={test_f1_weighted:.4f}"
            )
            if test_top2_acc is not None:
                logger.info(f"[TEST Top-2 Accuracy] {test_top2_acc:.4f}")
        
    if not cfg.fast_search:
        # Plotting
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('dark_background')
        plt.rcParams.update({"figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e"})

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        mode_suffix = f" | Binario: {selected_binary_mode}" if cfg.num_classes == 2 else ""
        fig.suptitle(
            f"Progreso de Entrenamiento (Ventana: {cfg.upto_week} | Clases: {cfg.num_classes}{mode_suffix})",
            fontsize=16,
            fontweight='bold',
            y=0.98,
        )

        train_loss_hist = np.array(history.history.get('loss', []), dtype=float)
        val_loss_hist = np.array(history.history.get('val_loss', []), dtype=float)
        train_acc_hist = np.array(history.history.get('accuracy', []), dtype=float)
        val_acc_hist = np.array(history.history.get('val_accuracy', []), dtype=float)
        val_bal_hist = np.array(history.history.get('val_balanced_acc', []), dtype=float)
        epochs_idx = np.arange(1, len(train_loss_hist) + 1)

        axes[0, 0].plot(epochs_idx, train_loss_hist, label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs_idx, val_loss_hist, label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Evolución de la Pérdida', pad=10)
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.3)

        axes[0, 1].plot(epochs_idx, train_acc_hist, label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs_idx, val_acc_hist, label='Validation Accuracy', linewidth=2)
        if len(val_bal_hist) == len(epochs_idx) and len(val_bal_hist) > 0:
            axes[0, 1].plot(epochs_idx, val_bal_hist, label='Validation Balanced Acc', linewidth=2, linestyle='--')
        if "top_2_acc" in history.history:
            axes[0, 1].plot(epochs_idx, history.history["top_2_acc"], label="Train Top-2", linewidth=1.8, linestyle=':')
        if "val_top_2_acc" in history.history:
            axes[0, 1].plot(epochs_idx, history.history["val_top_2_acc"], label="Val Top-2", linewidth=1.8, linestyle=':')
        axes[0, 1].set_title('Métricas de Clasificación', pad=10)
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)

        if len(train_loss_hist) == len(val_loss_hist) and len(train_acc_hist) == len(val_acc_hist):
            loss_gap = val_loss_hist - train_loss_hist
            acc_gap = val_acc_hist - train_acc_hist
            axes[1, 0].plot(epochs_idx, loss_gap, label='Gap Loss (val-train)', linewidth=2)
            axes[1, 0].plot(epochs_idx, acc_gap, label='Gap Accuracy (val-train)', linewidth=2)
            axes[1, 0].axhline(0.0, color='white', alpha=0.4, linestyle='--')
            axes[1, 0].set_title('Gap de Generalización', pad=10)
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Gap')
            axes[1, 0].legend()
            axes[1, 0].grid(True, linestyle='--', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No hay series suficientes para gap', ha='center', va='center')
            axes[1, 0].set_title('Gap de Generalización', pad=10)
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        lr_hist = history.history.get('learning_rate', history.history.get('lr', []))
        if len(lr_hist) == len(epochs_idx) and len(lr_hist) > 0:
            axes[1, 1].plot(epochs_idx, lr_hist, label='Learning Rate', linewidth=2, color='#f39c12')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_title('Evolución del Learning Rate', pad=10)
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('LR (log)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle='--', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.6, f"Val Acc final: {float(results[1]):.4f}", ha='center', va='center')
            axes[1, 1].text(0.5, 0.45, f"Val Balanced Acc: {val_bal_acc:.4f}", ha='center', va='center')
            if 'auc_val' in locals():
                axes[1, 1].text(0.5, 0.3, f"Val AUC OVR: {float(auc_val):.4f}", ha='center', va='center')
            axes[1, 1].set_title('Resumen de Rendimiento', pad=10)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        sns.despine(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = save_dir / f"plot_uptoW{cfg.upto_week}_{target_tag}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Gráfico guardado en: {plot_path}")

        fig_diag, ax_diag = plt.subplots(1, 2, figsize=(16, 6))
        fig_diag.suptitle(
            f"Diagnóstico Validación (Ventana: {cfg.upto_week} | Clases: {cfg.num_classes}{mode_suffix})",
            fontsize=14,
            fontweight='bold',
        )

        cm_norm = confusion_matrix(y_val, y_pred, normalize='true')
        sns.heatmap(
            cm_norm,
            ax=ax_diag[0],
            cmap='mako',
            annot=True,
            fmt='.2f',
            cbar=True,
            vmin=0.0,
            vmax=1.0,
        )
        ax_diag[0].set_title('Matriz de Confusión Normalizada (VAL)')
        ax_diag[0].set_xlabel('Predicción')
        ax_diag[0].set_ylabel('Clase real')

        if cfg.num_classes == 2:
            pos_scores = y_probs[:, 1]
            ax_diag[1].hist(pos_scores[y_val == 0], bins=30, alpha=0.6, label='Real clase 0', density=True)
            ax_diag[1].hist(pos_scores[y_val == 1], bins=30, alpha=0.6, label='Real clase 1', density=True)
            thr = float(selected_threshold) if selected_threshold is not None else 0.5
            ax_diag[1].axvline(thr, color='white', linestyle='--', alpha=0.8, label=f'Threshold={thr:.2f}')
            ax_diag[1].set_title('Distribución de Probabilidad Clase Positiva (VAL)')
            ax_diag[1].set_xlabel('p(clase positiva)')
            ax_diag[1].set_ylabel('Densidad')
            ax_diag[1].legend()
            ax_diag[1].grid(True, linestyle='--', alpha=0.25)
        else:
            conf_max = np.max(y_probs, axis=1)
            is_ok = (y_pred == y_val)
            ax_diag[1].hist(conf_max[is_ok], bins=30, alpha=0.6, label='Predicciones correctas', density=True)
            ax_diag[1].hist(conf_max[~is_ok], bins=30, alpha=0.6, label='Predicciones erróneas', density=True)
            ax_diag[1].set_title('Confianza Máxima en Predicción (VAL)')
            ax_diag[1].set_xlabel('max softmax')
            ax_diag[1].set_ylabel('Densidad')
            ax_diag[1].legend()
            ax_diag[1].grid(True, linestyle='--', alpha=0.25)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        diag_path = save_dir / f"diagnostics_val_uptoW{cfg.upto_week}_{target_tag}.png"
        plt.savefig(diag_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Diagnóstico VAL guardado en: {diag_path}")
        
        # Guardar modelo entrenado
        model_dir = Path("/workspace/TFM_education_ai_analytics/models/transformers")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"transformer_uptoW{cfg.upto_week}_{target_tag}.keras"
        model.save(model_path)
        logger.info(f"✅ Modelo final guardado en: {model_path}")
    

    # Guardar métricas y configuración en historial via hyperparams
    test_loss = float(results_test[0]) if cfg.eval_test else None
    test_acc = float(results_test[1]) if cfg.eval_test else None
    test_bacc = float(test_bal_acc_val) if cfg.eval_test else None
    test_auc_val = float(auc_test) if cfg.eval_test and 'auc_test' in locals() else None

    if not cfg.eval_test:
        test_precision = test_recall = test_f1 = None
        test_precision_macro = test_recall_macro = test_f1_macro = None
        test_precision_weighted = test_recall_weighted = test_f1_weighted = None
        test_top2_acc = None

    # Payload explícito para automatización (Random Search, Optuna, etc.)
    val_metrics = {
        "val_loss": float(results[0]),
        "val_accuracy": float(results[1]),
        "val_balanced_acc": val_bal_acc,
        "val_f1": float(val_f1 if 'val_f1' in locals() and val_f1 is not None else 0),
        "val_recall": float(val_recall if 'val_recall' in locals() and val_recall is not None else 0),
        "val_precision": float(val_precision if 'val_precision' in locals() and val_precision is not None else 0),
        "val_auc": float(auc_val) if 'auc_val' in locals() else 0,
        "val_precision_macro": float(val_precision_macro) if val_precision_macro is not None else None,
        "val_recall_macro": float(val_recall_macro) if val_recall_macro is not None else None,
        "val_f1_macro": float(val_f1_macro) if val_f1_macro is not None else None,
        "val_precision_weighted": float(val_precision_weighted) if val_precision_weighted is not None else None,
        "val_recall_weighted": float(val_recall_weighted) if val_recall_weighted is not None else None,
        "val_f1_weighted": float(val_f1_weighted) if val_f1_weighted is not None else None,
        "val_top2_acc": float(val_top2_acc) if val_top2_acc is not None else None,
        "val_threshold": float(selected_threshold) if selected_threshold is not None else None,
    }
    logger.info("VAL_METRICS: " + json.dumps(val_metrics))

    if not cfg.fast_search:
        hp.save_experiment(
            save_dir=save_dir,
            upto_week=cfg.upto_week,
            paper_baseline=cfg.paper_baseline,
            binary_mode=selected_binary_mode,
            val_loss=results[0],
            val_acc=results[1],
            val_balanced_acc=val_bal_acc,
            val_auc=float(auc_val) if 'auc_val' in locals() else None,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            val_precision_macro=val_precision_macro,
            val_recall_macro=val_recall_macro,
            val_f1_macro=val_f1_macro,
            val_precision_weighted=val_precision_weighted,
            val_recall_weighted=val_recall_weighted,
            val_f1_weighted=val_f1_weighted,
            val_top2_acc=val_top2_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            test_balanced_acc=test_bacc,
            test_auc=test_auc_val,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1=test_f1,
            test_precision_macro=test_precision_macro,
            test_recall_macro=test_recall_macro,
            test_f1_macro=test_f1_macro,
            test_precision_weighted=test_precision_weighted,
            test_recall_weighted=test_recall_weighted,
            test_f1_weighted=test_f1_weighted,
            test_top2_acc=test_top2_acc,
            history_filename=cfg.history_filename
        )
        
        history_file_used = cfg.history_filename if cfg.history_filename else f"experiments_history_{target_tag}.json"
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
                "test_precision_macro": test_precision_macro,
                "test_recall_macro": test_recall_macro,
                "test_f1_macro": test_f1_macro,
                "test_precision_weighted": test_precision_weighted,
                "test_recall_weighted": test_recall_weighted,
                "test_f1_weighted": test_f1_weighted,
                "test_top2_acc": test_top2_acc,
            },
            "selected_threshold": float(selected_threshold) if selected_threshold is not None else None,
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"💾 Metrics JSON guardado en: {metrics_out}")

if __name__ == "__main__":
    app()
