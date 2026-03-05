from dataclasses import dataclass, field
from typing import List

from educational_ai_analytics.config import N_CLUSTERS

@dataclass
class AutoencoderParams:
    """Configuración para el modelo StudentProfileAutoencoder (Arquitectura Pro)."""
    input_dim: int = 60
    latent_dim: int = 16   # ✅ cuello de botella real (era 32 → mismo tamaño que último hidden)
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])  # ✅ encoder más profundo
    dropout_rate: float = 0.01
    denoise_std: float = 0.01
    l2_latent: float = 1e-6
    z_norm_penalty: float = 1e-4
    normalize_latent: bool = False  # Desactivar para que el clustering sea más natural
    activation: str = "leaky_relu"
    
    # Entrenamiento por fases
    pretrain_epochs: int = 40
    joint_epochs: int = 40
    batch_size: int = 254
    # RTX 50xx + current TF nightly can hit PTX compiler crashes with mixed precision.
    use_mixed_precision: bool = False
    learning_rate: float = 0.001

    # ✅ NEW: Configuración para Deep Clustering (DCN)
    n_clusters: int = N_CLUSTERS
    use_clustering_objective: bool = True

    # 🔧 FIX #2 — reducir peso KL para evitar overfitting en clustering
    clustering_loss_weight: float = 0.3    # ✅ era 0.05 → más presión de clustering (espacio latente realmente se remodela)
    clustering_loss_scale: float = 1.0

    # 🔧 FIX #2 — incluir val en el cálculo de P (ver train_autoencoder.py)
    sample_frac: float = 1.0

    # 🔧 FIX #3 — target_blend bajo al inicio, warmup lineal en train_autoencoder
    target_blend: float = 0.05            # era 0.2 → arranca casi como soft-Q
    target_blend_max: float = 0.35        # techo del warmup
    target_blend_warmup_epochs: int = 20  # épocas hasta alcanzar target_blend_max

    # 🔧 FIX #3 — LR más conservador en fase 3 y gradient clipping
    lr_phase3_divisor: float = 5.0        # ✅ era 20.0 → LR/5 (2e-4) para que joint training realmente actualice pesos
    grad_clip_norm: float = 1.0           # gradient clipping global

    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 0.00001

# Instancia por defecto
AE_PARAMS = AutoencoderParams()
