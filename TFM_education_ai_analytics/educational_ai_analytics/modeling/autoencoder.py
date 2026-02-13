import tensorflow as tf
from tensorflow.keras import layers

class StudentProfileAutoencoder(tf.keras.Model):
    """
    Autoencoder denso avanzado para perfiles de estudiantes.
    Incluye BatchNormalization y LeakyReLU para maximizar la convergencia.
    """
    def __init__(self, 
                 input_dim=61,
                 latent_dim=24, 
                 hidden_dims=[64, 48, 32],
                 dropout_rate=0.1,
                 **kwargs):
        super(StudentProfileAutoencoder, self).__init__(**kwargs)
        self.input_dim = input_dim

        # 1. ENCODER
        self.encoder_layers = tf.keras.Sequential()
        for dims in hidden_dims:
            self.encoder_layers.add(layers.Dense(dims))
            self.encoder_layers.add(layers.BatchNormalization())
            self.encoder_layers.add(layers.LeakyReLU(negative_slope=0.1))
            self.encoder_layers.add(layers.Dropout(dropout_rate))
        
        # Espacio Latente
        self.latent_layer = layers.Dense(latent_dim, name="latent_space")
        self.latent_bn = layers.BatchNormalization()
        self.latent_act = layers.LeakyReLU(negative_slope=0.1)

        # 2. DECODER
        self.decoder_layers = tf.keras.Sequential()
        for dims in reversed(hidden_dims):
            self.decoder_layers.add(layers.Dense(dims))
            self.decoder_layers.add(layers.BatchNormalization())
            self.decoder_layers.add(layers.LeakyReLU(negative_slope=0.1))
            self.decoder_layers.add(layers.Dropout(dropout_rate))
        
        # Capa de salida: Reconstrucción lineal
        self.output_layer = layers.Dense(input_dim, activation='linear')

    def encode(self, x):
        """Mapea la entrada al espacio latente con normalización."""
        x = self.encoder_layers(x)
        x = self.latent_layer(x)
        x = self.latent_bn(x)
        return self.latent_act(x)
    
    def decode(self, latent_vector):
        """Reconstruye el perfil a partir del vector latente."""
        x = self.decoder_layers(latent_vector)
        return self.output_layer(x)

    def call(self, inputs):
        """Flujo completo del Autoencoder."""
        latent = self.encode(inputs)
        return self.decode(latent)