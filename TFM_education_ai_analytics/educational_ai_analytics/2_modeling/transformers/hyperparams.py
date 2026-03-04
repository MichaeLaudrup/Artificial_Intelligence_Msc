import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TransformerHyperparams:
    # General
    upto_week: int = 5
    with_static: bool = True
    use_clustering_features: bool = True
    accumulated_uptow: bool = True
    batch_size: int = 64
    epochs: int = 80

    # Arquitectura Temporal (Transformer)
    latent_d: int = 256
    ff_dim: int = 128
    dropout: float = 0.3
    
    #Cosas que afecta a la capa temporal
    # Cuantos puntos de vista 
    num_heads: int = 8
    num_layers: int = 2
    
    # Arquitectura Estática
    static_hidden: List[int] = field(default_factory=lambda: [ 64, 32])

    # Capas de la red neuronal que fusiona la info temporal y estatica
    head_hidden: List[int] = field(default_factory=lambda: [256, 128,64,32 ])
    
    # Optimizador
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    
    # Callbacks
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 7
    reduce_lr_min_lr: float = 1e-6
    early_stopping_patience: int = 10

    # Focal Loss
    focal_gamma: float = 2.5

    # Binarización para num_classes == 2
    # - "paper": Pass/Dist vs Withdrawn (excluye Fail)
    # - "original": Pass/Dist vs Fail (excluye Withdrawn)
    # - "success_vs_risk": Pass/Dist vs Fail/Withdrawn
    binary_mode: str = "paper"


    num_classes: int = 2
    # [Peso Clase 0 (No Riesgo), Peso Clase 1 (Riesgo)] 
    # cuidadito añadir clases implica añadir elementos aqui
    # PROBLEMA 2 CLASES: [0,1] -> [0.27, 0.73] [Pass , Withdraw]
    # PROBLEMA 2 CLASES ALTERNATIVO (success_vs_risk): [0,1] -> [0.46, 0.54] [Pass/Distinction vs Fail/Withdraw]
    # PROBLEMA 3 CLASES: [0,1,2] -> [0.43, 0.37, 0.2] [Fail, Withdraw, Pass]
    # PROBLEMA 4 CLASES: [0,1,2,3] -> [0.27, 0.35, 0.15, 0.23] [Fail, Withdraw, Pass, Distinction]
    # focal_alpha: List[float] = field(default_factory=lambda: [0.35, 0.35, 0.2, 0.1])
    focal_alpha: List[float] = field(default_factory=lambda:  [0.27, 0.73])

    def save_experiment(
        self,
        save_dir: Path,
        upto_week: int,
        paper_baseline: bool,
        binary_mode: Optional[str],
        val_loss: float,
        val_acc: float,
        val_balanced_acc: float,
        val_auc: Optional[float],
        val_precision: Optional[float] = None,
        val_recall: Optional[float] = None,
        val_f1: Optional[float] = None,
        val_precision_macro: Optional[float] = None,
        val_recall_macro: Optional[float] = None,
        val_f1_macro: Optional[float] = None,
        val_precision_weighted: Optional[float] = None,
        val_recall_weighted: Optional[float] = None,
        val_f1_weighted: Optional[float] = None,
        val_top2_acc: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        test_balanced_acc: Optional[float] = None,
        test_auc: Optional[float] = None,
        test_precision: Optional[float] = None,
        test_recall: Optional[float] = None,
        test_f1: Optional[float] = None,
        test_precision_macro: Optional[float] = None,
        test_recall_macro: Optional[float] = None,
        test_f1_macro: Optional[float] = None,
        test_precision_weighted: Optional[float] = None,
        test_recall_weighted: Optional[float] = None,
        test_f1_weighted: Optional[float] = None,
        test_top2_acc: Optional[float] = None,
        history_filename: Optional[str] = None
    ):
        if history_filename:
            history_path = save_dir / history_filename
        else:
            if self.num_classes == 2:
                mode = str(binary_mode).strip().lower() if binary_mode else ("paper" if paper_baseline else "original")
                history_path = save_dir / f"experiments_history_{self.num_classes}clases_{mode}.json"
            else:
                history_path = save_dir / f"experiments_history_{self.num_classes}clases.json"
        
        results_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": {
                "upto_week": upto_week,
                "num_classes": self.num_classes,
                "paper_baseline": paper_baseline,
                "binary_mode": binary_mode,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "with_static": self.with_static,
                "use_clustering_features": self.use_clustering_features,
                   "accumulated_uptow": self.accumulated_uptow,
                "latent_d": self.latent_d,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "num_layers": self.num_layers,
                "static_hidden": self.static_hidden,
                "head_hidden": self.head_hidden,
                "learning_rate": self.learning_rate,
                "clipnorm": self.clipnorm,
                "reduce_lr_factor": self.reduce_lr_factor,
                "reduce_lr_patience": self.reduce_lr_patience,
                "reduce_lr_min_lr": self.reduce_lr_min_lr,
                "early_stopping_patience": self.early_stopping_patience,
                "focal_gamma": self.focal_gamma,
                "focal_alpha": self.focal_alpha
            },
            "validation_metrics": {
                "loss": float(val_loss),
                "accuracy": float(val_acc),
                "balanced_accuracy": float(val_balanced_acc),
                "auc_ovr": float(val_auc) if val_auc is not None else None
            }
        }
        
        if val_precision is not None: results_payload["validation_metrics"]["precision"] = float(val_precision)
        if val_recall is not None: results_payload["validation_metrics"]["recall"] = float(val_recall)
        if val_f1 is not None: results_payload["validation_metrics"]["f1_score"] = float(val_f1)
        if val_precision_macro is not None: results_payload["validation_metrics"]["precision_macro"] = float(val_precision_macro)
        if val_recall_macro is not None: results_payload["validation_metrics"]["recall_macro"] = float(val_recall_macro)
        if val_f1_macro is not None: results_payload["validation_metrics"]["f1_macro"] = float(val_f1_macro)
        if val_precision_weighted is not None: results_payload["validation_metrics"]["precision_weighted"] = float(val_precision_weighted)
        if val_recall_weighted is not None: results_payload["validation_metrics"]["recall_weighted"] = float(val_recall_weighted)
        if val_f1_weighted is not None: results_payload["validation_metrics"]["f1_weighted"] = float(val_f1_weighted)
        if val_top2_acc is not None: results_payload["validation_metrics"]["top_2_acc"] = float(val_top2_acc)
        
        if test_loss is not None:
            results_payload["test_metrics"] = {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "balanced_accuracy": float(test_balanced_acc),
                "auc_ovr": float(test_auc) if test_auc is not None else None
            }
            if test_precision is not None: results_payload["test_metrics"]["precision"] = float(test_precision)
            if test_recall is not None: results_payload["test_metrics"]["recall"] = float(test_recall)
            if test_f1 is not None: results_payload["test_metrics"]["f1_score"] = float(test_f1)
            if test_precision_macro is not None: results_payload["test_metrics"]["precision_macro"] = float(test_precision_macro)
            if test_recall_macro is not None: results_payload["test_metrics"]["recall_macro"] = float(test_recall_macro)
            if test_f1_macro is not None: results_payload["test_metrics"]["f1_macro"] = float(test_f1_macro)
            if test_precision_weighted is not None: results_payload["test_metrics"]["precision_weighted"] = float(test_precision_weighted)
            if test_recall_weighted is not None: results_payload["test_metrics"]["recall_weighted"] = float(test_recall_weighted)
            if test_f1_weighted is not None: results_payload["test_metrics"]["f1_weighted"] = float(test_f1_weighted)
            if test_top2_acc is not None: results_payload["test_metrics"]["top_2_acc"] = float(test_top2_acc)
            
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []
            
        history.append(results_payload)
            
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)


TRANSFORMER_PARAMS = TransformerHyperparams()