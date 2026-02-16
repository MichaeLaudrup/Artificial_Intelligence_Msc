from dataclasses import dataclass, field
from typing import List

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
CLUSTER_PARAMS = ClusteringParams()
GLOBAL_TRAIN_PARAMS = TrainingParams()
