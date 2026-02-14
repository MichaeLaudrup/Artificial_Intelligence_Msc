from dataclasses import dataclass, field
from typing import List

@dataclass
class AutoencoderParams:
    """Configuración para el modelo StudentProfileAutoencoder (Arquitectura Pro)."""
    input_dim: int = 43  
    latent_dim: int = 24
    hidden_dims: List[int] = field(default_factory=lambda: [32])
    dropout_rate: float = 0.03
    denoise_std: float = 0.03
    l2_latent: float = 1e-5
    z_norm_penalty: float = 1e-4
    activation: str = "leaky_relu"
    
    # Entrenamiento
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    
    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 0.00001

@dataclass
class ClusteringParams:
    """Configuración para los algoritmos de clustering."""
    k_means_clusters: int = 4
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

@dataclass
class TrainingParams:
    """Parámetros generales de entrenamiento."""
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15

# Instancias por defecto para usar en todo el proyecto
AE_PARAMS = AutoencoderParams()
CLUSTER_PARAMS = ClusteringParams()
GLOBAL_TRAIN_PARAMS = TrainingParams()
