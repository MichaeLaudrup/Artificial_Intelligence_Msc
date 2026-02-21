import tensorflow as tf
from tensorflow.keras import layers

# =========================
# Positional Encoding (sin/cos intercalado)
# =========================
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 512, name=None):
        super().__init__(name=name)
        if d_model % 2 != 0:
            raise ValueError("d_model debe ser par para positional encoding sinusoidal estándar")

        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]  # (L,1)
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]    # (1,D)

        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates  # (L,D)

        # Intercalado sin/cos: [sin0, cos0, sin1, cos1, ...]
        sin = tf.sin(angle_rads[:, 0::2])  # (L, D/2)
        cos = tf.cos(angle_rads[:, 1::2])  # (L, D/2)
        pe = tf.reshape(tf.stack([sin, cos], axis=-1), (max_len, d_model))  # (L,D)

        self.pe = pe[tf.newaxis, ...]  # (1,L,D)

    def call(self, x):
        # x: (B,W,D)
        w = tf.shape(x)[1]
        return x + self.pe[:, :w, :]


# =========================
# GLU (Gated Linear Unit)
# =========================
class GLULayer(layers.Layer):
    """Dense(2*units) -> split -> a * sigmoid(b)"""
    def __init__(self, units: int, name=None):
        super().__init__(name=name)
        self.proj = layers.Dense(2 * units)

    def call(self, x):
        z = self.proj(x)
        a, b = tf.split(z, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)


# =========================
# Encoder Block (Pre-LN + mask (B,W,W) + FFN GLU)
# =========================
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float, name=None):
        super().__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError("d_model debe ser divisible por num_heads")

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

        self.norm_attn = layers.LayerNormalization(epsilon=1e-6)
        self.norm_ffn  = layers.LayerNormalization(epsilon=1e-6)

        self.drop_attn = layers.Dropout(dropout)
        self.drop_ffn  = layers.Dropout(dropout)

        # FFN: d_model -> ff_dim (con gating GLU) -> d_model  <---- (Reemplaza a ReLU)
        self.ffn_glu = GLULayer(units=ff_dim)
        self.ffn_out = layers.Dense(d_model)

    @staticmethod
    def make_attn_mask(seq_mask: tf.Tensor) -> tf.Tensor:
        """
        seq_mask: (B,W) int/bool, 1=valid, 0=pad
        return: (B,W,W) bool
        """
        sm = tf.cast(seq_mask, tf.bool)
        return sm[:, tf.newaxis, :] & sm[:, :, tf.newaxis]

    def call(self, x, training=False, seq_mask=None):
        if seq_mask is None:
            raise ValueError("seq_mask es obligatorio (B,W).")

        attn_mask = self.make_attn_mask(seq_mask)

        # --- Self-attention (Pre-LN) ---
        h = self.norm_attn(x)
        attn = self.mha(h, h, h, attention_mask=attn_mask, training=training)
        attn = self.drop_attn(attn, training=training)
        x = x + attn

        # --- FFN GLU (Pre-LN) ---
        h = self.norm_ffn(x)
        ffn = self.ffn_glu(h)
        ffn = self.ffn_out(ffn)
        ffn = self.drop_ffn(ffn, training=training)
        x = x + ffn

        return x


# =========================
# Modelo completo: Temporal (Transformer) + Static (cluster feats) + Head MLP
# =========================
class GLUTransformerClassifier(tf.keras.Model):
    def __init__(
        self,
        latent_d: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_classes: int,
        num_layers: int = 2,
        max_len: int = 512,
        with_static_features: bool = True,
        static_hidden=(64, 64),
        head_hidden=(128, 64),
        name=None,
    ):
        super().__init__(name=name)
        self.with_static_features = with_static_features

        # ----- Temporal -----
        self.input_proj = layers.Dense(latent_d)
        self.pos_encoding = PositionalEncoding(d_model=latent_d, max_len=max_len)
        self.in_drop = layers.Dropout(dropout)

        self.encoders = [
            TransformerEncoderBlock(latent_d, num_heads, ff_dim, dropout, name=f"enc_{i}")
            for i in range(num_layers)
        ]

        # Normaliza antes de hacer el pooling promedio
        self.seq_out_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pooled_norm = layers.LayerNormalization(epsilon=1e-6)

        # ----- Static -----
        if self.with_static_features:
            self.static_block = tf.keras.Sequential(
                [layers.Dense(h, activation="relu") for h in static_hidden] +
                [layers.LayerNormalization(epsilon=1e-6)],
                name="static_block"
            )
            self.fusion = layers.Concatenate(axis=-1)

        # ----- Head (MLP para Clasificación Final) -----
        head_layers = []
        for h in head_hidden:
            head_layers += [
                layers.Dropout(dropout),
                layers.Dense(h, activation="relu"),
                layers.LayerNormalization(epsilon=1e-6),
            ]
        head_layers += [
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
        self.head = tf.keras.Sequential(head_layers, name="head")

    def call(self, inputs, training=False):
        # inputs: (x_seq, seq_mask, x_static) o (x_seq, seq_mask)
        if not isinstance(inputs, (tuple, list)):
            raise ValueError("inputs debe ser tupla/lista: (x_seq, seq_mask[, x_static])")

        if self.with_static_features:
            if len(inputs) != 3:
                raise ValueError("Con static features: inputs=(x_seq, seq_mask, x_static)")
            x_seq, seq_mask, x_static = inputs
        else:
            if len(inputs) != 2:
                raise ValueError("Sin static features: inputs=(x_seq, seq_mask)")
            x_seq, seq_mask = inputs
            x_static = None

        if seq_mask is None:
            raise ValueError("seq_mask es obligatorio (B,W).")

        # ----- Static path -----
        if self.with_static_features:
            x_static = self.static_block(x_static, training=training)

        # ----- Temporal path -----
        x_seq = self.input_proj(x_seq)          # (B,W,D)
        x_seq = self.pos_encoding(x_seq)        # (B,W,D)
        x_seq = self.in_drop(x_seq, training=training)

        for enc in self.encoders:
            x_seq = enc(x_seq, training=training, seq_mask=seq_mask)

        x_seq = self.seq_out_norm(x_seq)

        # ----- Masked mean pooling -----
        # Mapea los pesos a cero ahí donde no hay secuencia real
        m = tf.cast(seq_mask, x_seq.dtype)[:, :, tf.newaxis]  # (B,W,1)
        pooled = tf.reduce_sum(x_seq * m, axis=1) / (tf.reduce_sum(m, axis=1) + 1e-8)
        pooled = self.pooled_norm(pooled)

        # ----- Fusion -----
        if self.with_static_features:
            z = self.fusion([pooled, x_static])
        else:
            z = pooled

        # Clasificación final
        return self.head(z, training=training)
