from dataclasses import dataclass, field
from typing import List

@dataclass
class AutoencoderParams:
    """Configuración para el modelo StudentProfileAutoencoder (Arquitectura Pro)."""
    input_dim: int = 60  
    latent_dim: int = 28
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.01
    denoise_std: float = 0.01
    l2_latent: float = 1e-6
    z_norm_penalty: float = 1e-4
    normalize_latent: bool = False  # Desactivar para que el clustering sea más natural
    activation: str = "leaky_relu"
    
    # Entrenamiento por fases
    pretrain_epochs: int = 25
    joint_epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.001

    # ✅ NEW: Configuración para Deep Clustering (DCN)
    n_clusters: int = 5
    clustering_loss_weight: float = 0.2  # Importancia que le damos a intentar hacer clusters más definidos
    sample_frac: float = 1.0              # Fracción de datos para calcular el target P (0.1 a 1.0)
    
    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 0.00001

# Instancia por defecto
AE_PARAMS = AutoencoderParams()
