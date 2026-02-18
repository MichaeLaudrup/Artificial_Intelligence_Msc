import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
from .hyperparams import AE_PARAMS


@tf.keras.utils.register_keras_serializable(package="edu")
class ClusteringLayer(layers.Layer):
    """
    DEC/IDEC-style clustering layer using Student's t-distribution kernel.

    Given embeddings z (B, D) and trainable cluster centers mu (K, D),
    returns soft assignments q (B, K).

    q_ij ∝ (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
    """
    def __init__(self, n_clusters: int, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)

    def build(self, input_shape):
        # input_shape: (batch, latent_dim)
        if len(input_shape) != 2:
            raise ValueError(f"ClusteringLayer expects rank-2 input (B, D). Got: {input_shape}")

        latent_dim = int(input_shape[-1])
        self.clusters = self.add_weight(
            name="clusters",
            shape=(self.n_clusters, latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, D)
        z = tf.expand_dims(inputs, axis=1)           # (B, 1, D)
        mu = tf.expand_dims(self.clusters, axis=0)   # (1, K, D)
        dist = tf.reduce_sum(tf.square(z - mu), axis=2)  # (B, K)

        q = 1.0 / (1.0 + dist / self.alpha)
        q = tf.pow(q, (self.alpha + 1.0) / 2.0)

        # Normalize row-wise to get probabilities
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)

        return q

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "alpha": self.alpha,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="edu")
class StudentProfileAutoencoder(tf.keras.Model):
    """
    Autoencoder + DEC clustering head.

    - Encoder: MLP (deep) + linear residual (wide) + per-dimension learned gate
    - Latent regularization: optional L2 norm penalty + optional L2-normalization
    - Outputs:
        1) reconstruction (B, input_dim)  [name: "reconstruction"]
        2) soft cluster assignments q (B, n_clusters) [layer name: "clustering_output"]
    """

    def __init__(
        self,
        input_dim: int = AE_PARAMS.input_dim,
        latent_dim: int = AE_PARAMS.latent_dim,
        hidden_dims=AE_PARAMS.hidden_dims,
        n_clusters: int = AE_PARAMS.n_clusters if hasattr(AE_PARAMS, "n_clusters") else 6,
        dropout_rate: float = AE_PARAMS.dropout_rate,
        denoise_std: float = AE_PARAMS.denoise_std,
        l2_latent: float = AE_PARAMS.l2_latent,
        z_norm_penalty: float = AE_PARAMS.z_norm_penalty,
        normalize_latent: bool = AE_PARAMS.normalize_latent,
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_clusters = int(n_clusters)

        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = float(dropout_rate)
        self.denoise_std = float(denoise_std)
        self.l2_latent = float(l2_latent)
        self.z_norm_penalty = float(z_norm_penalty)
        self.normalize_latent = bool(normalize_latent)
        self.alpha = float(alpha)

        # -----------------------
        # Encoder
        # -----------------------
        enc = []
        for dims in self.hidden_dims:
            enc.append(layers.Dense(int(dims)))
            enc.append(layers.BatchNormalization())
            enc.append(layers.LeakyReLU(0.1))
            if self.dropout_rate and self.dropout_rate > 0:
                enc.append(layers.Dropout(self.dropout_rate))
        self.encoder_layers = tf.keras.Sequential(enc, name="encoder")

        # Latent projections
        self.latent_deep = layers.Dense(
            self.latent_dim,
            name="latent_deep",
            kernel_regularizer=regularizers.l2(self.l2_latent),
        )

        self.latent_residual = layers.Dense(
            self.latent_dim,
            use_bias=False,
            name="latent_residual",
            kernel_regularizer=regularizers.l2(self.l2_latent),
        )

        self.gate = layers.Dense(
            self.latent_dim,
            activation="sigmoid",
            name="latent_gate",
        )

        # Clustering head
        self.clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            name="clustering_output",
        )

        # -----------------------
        # Decoder
        # -----------------------
        dec = []
        for dims in reversed(tuple(self.hidden_dims)):
            dec.append(layers.Dense(int(dims)))
            dec.append(layers.BatchNormalization())
            dec.append(layers.LeakyReLU(0.1))
            if self.dropout_rate and self.dropout_rate > 0:
                dec.append(layers.Dropout(self.dropout_rate))
        self.decoder_layers = tf.keras.Sequential(dec, name="decoder")

        self.output_layer = layers.Dense(
            self.input_dim,
            activation="linear",
            name="reconstruction",
        )

    def encode(self, x, training: bool = False):
        # Denoising only during training
        if training and self.denoise_std > 0:
            noise = tf.random.normal(tf.shape(x), stddev=self.denoise_std, dtype=x.dtype)
            x = x + noise

        x_deep = self.encoder_layers(x, training=training)
        z_deep = self.latent_deep(x_deep)
        z_linear = self.latent_residual(x)

        g = self.gate(x_deep)
        z = g * z_deep + (1.0 - g) * z_linear

        # Soft L2 norm penalty (optional)
        if training and self.z_norm_penalty > 0:
            penalty = self.z_norm_penalty * tf.reduce_mean(tf.reduce_sum(tf.square(z), axis=1))
            self.add_loss(tf.cast(penalty, tf.float32))

        # Optional L2 normalization (unit sphere)
        if self.normalize_latent:
            z = tf.math.l2_normalize(z, axis=1)

        return z

    def decode(self, z, training: bool = False):
        x = self.decoder_layers(z, training=training)
        return self.output_layer(x)

    def call(self, inputs, training: bool = False):
        z = self.encode(inputs, training=training)
        x_recon = self.decode(z, training=training)
        q = self.clustering_layer(z)
        return x_recon, q

    def get_embeddings(self, x, batch_size: int = 1024):
        # Convert to tensor to avoid retracing & improve throughput
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices(x).batch(int(batch_size))
        zs = []
        for xb in ds:
            z = self.encode(xb, training=False)
            zs.append(z)
        return tf.concat(zs, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "n_clusters": self.n_clusters,
            "dropout_rate": self.dropout_rate,
            "denoise_std": self.denoise_std,
            "l2_latent": self.l2_latent,
            "z_norm_penalty": self.z_norm_penalty,
            "normalize_latent": self.normalize_latent,
            "alpha": self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
