import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TransformerHyperparams:
    # General
    num_classes: int = 2
    with_static: bool = True
    batch_size: int = 256
    epochs: int = 80

    # Arquitectura Temporal (Transformer)
    latent_d: int = 64
    num_heads: int = 2
    ff_dim: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    
    # Arquitectura Estática y Head
    static_hidden: List[int] = field(default_factory=lambda: [64, 32])
    head_hidden: List[int] = field(default_factory=lambda: [64, 32])
    
    # Optimizador
    learning_rate: float = 1e-3
    clipnorm: float = 1.0
    
    # Callbacks
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 3
    reduce_lr_min_lr: float = 1e-6
    early_stopping_patience: int = 10

    def save_experiment(
        self,
        save_dir: Path,
        upto_week: int,
        paper_baseline: bool,
        val_loss: float,
        val_acc: float,
        val_balanced_acc: float,
        val_auc: Optional[float],
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        test_balanced_acc: Optional[float] = None,
        test_auc: Optional[float] = None
    ):
        history_path = save_dir / "experiments_history.json"
        
        results_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": {
                "upto_week": upto_week,
                "num_classes": self.num_classes,
                "paper_baseline": paper_baseline,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "with_static": self.with_static,
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
                "early_stopping_patience": self.early_stopping_patience
            },
            "validation_metrics": {
                "loss": float(val_loss),
                "accuracy": float(val_acc),
                "balanced_accuracy": float(val_balanced_acc),
                "auc_ovr": float(val_auc) if val_auc is not None else None
            }
        }
        
        if test_loss is not None:
            results_payload["test_metrics"] = {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "balanced_accuracy": float(test_balanced_acc),
                "auc_ovr": float(test_auc) if test_auc is not None else None
            }
            
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