import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]  # (L, 1)
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]    # (1, D)

        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates

        # even -> sin, odd -> cos
        sin_part = tf.sin(angle_rads[:, 0::2])
        cos_part = tf.cos(angle_rads[:, 1::2])

        pe = tf.concat([sin_part, cos_part], axis=-1)
        pe = pe[tf.newaxis, ...]  # (1, L, D)
        self.pe = tf.cast(pe, tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.add1 = layers.Add()

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.add2 = layers.Add()

    def call(self, x, training=False, attention_mask=None):
        attn_out = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=attention_mask,
            training=training,
        )
        attn_out = self.dropout1(attn_out, training=training)
        x = self.add1([x, attn_out])
        x = self.norm1(x)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = self.add2([x, ffn_out])
        x = self.norm2(x)
        return x


class GLULayer(layers.Layer):
    # h(X) = (XW) ⊗ sigmoid(XV)
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = layers.Dense(2 * d_model)

    def call(self, x):
        z = self.proj(x)
        a, b = tf.split(z, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)


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
    ):
        super().__init__()

        if latent_d % num_heads != 0:
            raise ValueError("latent_d debe ser divisible por num_heads")

        self.input_proj = layers.Dense(latent_d)
        self.pos_encoding = PositionalEncoding(d_model=latent_d, max_len=max_len)
        self.input_dropout = layers.Dropout(dropout)

        self.encoders = [
            TransformerEncoderBlock(
                d_model=latent_d,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        self.glu = GLULayer(d_model=latent_d)
        self.norm_out = layers.LayerNormalization(epsilon=1e-6)

        self.head = tf.keras.Sequential(
            [
                layers.Dense(latent_d, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, inputs, training=False, mask=None):
        # inputs: (B, W, F)
        # mask:   (B, W) con 1/0 o bool (opcional)

        x = self.input_proj(inputs)          # (B, W, D)
        x = self.pos_encoding(x)             # (B, W, D)
        x = self.input_dropout(x, training=training)

        # Máscara para MHA: (B, 1, W), broadcast a (B, W, W)
        attn_mask = None
        if mask is not None:
            attn_mask = tf.cast(mask[:, tf.newaxis, :], tf.bool)

        for encoder in self.encoders:
            x = encoder(x, training=training, attention_mask=attn_mask)

        x = self.glu(x)
        x = self.norm_out(x)

        # Masked pooling temporal
        if mask is not None:
            m = tf.cast(mask, x.dtype)[:, :, tf.newaxis]     # (B, W, 1)
            x_sum = tf.reduce_sum(x * m, axis=1)             # (B, D)
            denom = tf.reduce_sum(m, axis=1) + 1e-8          # (B, 1)
            pooled = x_sum / denom
        else:
            pooled = tf.reduce_mean(x, axis=1)

        return self.head(pooled)
