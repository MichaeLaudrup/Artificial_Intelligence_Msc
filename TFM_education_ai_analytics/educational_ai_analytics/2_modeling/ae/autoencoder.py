import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
from .hyperparams import AE_PARAMS


class StudentProfileAutoencoder(tf.keras.Model):
    """
    Cambios vs tu versión:
    1) Mezcla Wide+Deep con GATE aprendible (por dimensión): z = g*z_deep + (1-g)*z_linear
       -> evita que uno de los dos caminos domine sin control.
    2) Denoising opcional (ruido gaussiano SOLO en training) en vez de depender del dropout "a ciegas".
    3) Regularización L2 en las proyecciones latentes + penalización suave de norma del embedding (opcional).
    4) Guardrails: clip del gate (vía sigmoid), y dropout más bajo por defecto.
    """

    def __init__(
        self,
        input_dim: int = AE_PARAMS.input_dim,
        latent_dim: int = AE_PARAMS.latent_dim,
        hidden_dims=AE_PARAMS.hidden_dims,
        dropout_rate: float = AE_PARAMS.dropout_rate,
        denoise_std: float = AE_PARAMS.denoise_std,
        l2_latent: float = AE_PARAMS.l2_latent,
        z_norm_penalty: float = AE_PARAMS.z_norm_penalty,
        normalize_latent: bool = AE_PARAMS.normalize_latent,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.denoise_std = float(denoise_std)
        self.z_norm_penalty = float(z_norm_penalty)
        self.normalize_latent = bool(normalize_latent)

        # -----------------------
        # Encoder
        # -----------------------
        enc = []
        for dims in hidden_dims:
            enc.append(layers.Dense(dims))
            enc.append(layers.BatchNormalization())
            enc.append(layers.LeakyReLU(0.1))
            if dropout_rate and dropout_rate > 0:
                enc.append(layers.Dropout(dropout_rate))
        self.encoder_layers = tf.keras.Sequential(enc, name="encoder")

        # Latent: Deep (no lineal)
        self.latent_deep = layers.Dense(
            latent_dim,
            name="latent_deep",
            kernel_regularizer=regularizers.l2(l2_latent),
        )

        # Latent: Residual (lineal / wide)
        self.latent_residual = layers.Dense(
            latent_dim,
            use_bias=False,
            name="latent_residual",
            kernel_regularizer=regularizers.l2(l2_latent),
        )

        # Gate por dimensión (0..1): decide cuánto usar deep vs linear
        self.gate = layers.Dense(
            latent_dim,
            activation="sigmoid",
            name="latent_gate",
        )

        # -----------------------
        # Decoder
        # -----------------------
        dec = []
        for dims in reversed(tuple(hidden_dims)):
            dec.append(layers.Dense(dims))
            dec.append(layers.BatchNormalization())
            dec.append(layers.LeakyReLU(0.1))
            if dropout_rate and dropout_rate > 0:
                dec.append(layers.Dropout(dropout_rate))
        self.decoder_layers = tf.keras.Sequential(dec, name="decoder")

        self.output_layer = layers.Dense(input_dim, activation="linear", name="reconstruction")

    def encode(self, x, training: bool = False):
        # Denoising: añade ruido SOLO en training (si denoise_std > 0)
        if training and self.denoise_std > 0:
            x = x + tf.random.normal(tf.shape(x), stddev=self.denoise_std)

        x_deep = self.encoder_layers(x, training=training)
        z_deep = self.latent_deep(x_deep)
        z_linear = self.latent_residual(x)

        # Mezcla con gate aprendido (por dimensión)
        g = self.gate(x_deep)
        z = g * z_deep + (1.0 - g) * z_linear

        # Penalización mucho más suave (o eliminada si z_norm_penalty=0)
        if training and self.z_norm_penalty > 0:
            self.add_loss(self.z_norm_penalty * tf.reduce_mean(tf.reduce_sum(tf.square(z), axis=1)))
        
        # Normalización L2 opcional (esfera unitaria)
        if self.normalize_latent:
            z = tf.math.l2_normalize(z, axis=1)

        return z

    def decode(self, z, training: bool = False):
        x = self.decoder_layers(z, training=training)
        return self.output_layer(x)

    def call(self, inputs, training: bool = False):
        z = self.encode(inputs, training=training)
        return self.decode(z, training=training)

    # Helper útil: embeddings directamente (para clustering)
    def get_embeddings(self, x, batch_size: int = 1024):
        # Convertir a tensor si es numpy para evitar problemas de re-tracing
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
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
            "denoise_std": self.denoise_std,
            "z_norm_penalty": self.z_norm_penalty,
            "normalize_latent": self.normalize_latent,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------
# Ejemplo de uso (opcional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy run
    ae = StudentProfileAutoencoder(input_dim=43, latent_dim=16, hidden_dims=(32,), denoise_std=0.03)

    # Compilación recomendada para tabular estandarizado
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))

    x = tf.random.normal((256, 43))
    ae.fit(x, x, epochs=2, batch_size=64, verbose=1)

    z = ae.get_embeddings(x)
    print("Embeddings:", z.shape)
