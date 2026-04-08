from dataclasses import dataclass, field
from typing import List

from educational_ai_analytics.config import N_CLUSTERS

@dataclass
class AutoencoderParams:
    """Configuración para el modelo StudentProfileAutoencoder (Arquitectura Pro)."""
    input_dim: int = 60
    latent_dim: int = 24
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.01
    denoise_std: float = 0.01
    l2_latent: float = 1e-6
    z_norm_penalty: float = 1e-4
    normalize_latent: bool = False 
    activation: str = "leaky_relu"
    
    
    pretrain_epochs: int = 100
    joint_epochs: int = 100
    batch_size: int = 254
    execution_device: str = "cpu"
    use_mixed_precision: bool = False
    learning_rate: float = 0.001

    n_clusters: int = N_CLUSTERS
    use_clustering_objective: bool = True

    clustering_loss_weight: float = 0.3
    clustering_loss_scale: float = 1.0

    sample_frac: float = 1.0

    target_blend: float = 0.05
    target_blend_max: float = 0.35        
    target_blend_warmup_epochs: int = 20  

    lr_phase3_divisor: float = 5.0
    grad_clip_norm: float = 1.0           

    
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 0.00001


AE_PARAMS = AutoencoderParams()
