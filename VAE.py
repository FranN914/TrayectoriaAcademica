import tensorflow as tf
import keras


class VAE(keras.Model):
    """
    Implementación de un Variational Autoencoder (VAE).
    Combina un encoder y un decoder, calcula las pérdidas de reconstrucción y KL-divergencia,
    y realiza el proceso de entrenamiento mediante optimización
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        Constructor de la clase VAE.
        Args:
            encoder: Modelo del codificador.
            decoder: Modelo del decodificador.
            **kwargs: Parámetros adicionales solicitados por `keras.Model`.
        """
        super().__init__(**kwargs)
        self.encoder = encoder  # Modelo del codificador
        self.decoder = decoder  # Modelo del decodificador

        # Métricas para el entrenamiento
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """
        Devuelve las métricas utilizadas durante el entrenamiento.
        Retorna:
        - Pérdida total (loss).
        - Pérdida de reconstrucción.
        - Pérdida KL-divergence.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Realiza un paso de entrenamiento personalizado.

        Args:
            data: Datos de entrada utilizados para el entrenamiento.

        Returns:
            Un diccionario con las métricas.
        """
        with tf.GradientTape() as tape:
            # Codificación: obtener la media, la varianza logarítmica y el vector z
            mean, log_var, z = self.encoder(data)

            # Restringir log_var para evitar valores extremos
            log_var = tf.clip_by_value(log_var, -5, 5)

            # Decodificación: reconstruir los datos a partir de la representación latente z
            reconstruction = self.decoder(z)
            # Calcular la pérdida de reconstrucción
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),  # Pérdida de reconstrucción
                )
            )
            # reconstruction_loss = tf.reduce_mean(
            #     keras.losses.binary_crossentropy(data, reconstruction)
            # )

            # )
            # Calcular la pérdida KL
            # (divergencia KL entre la distribución latente y una distribución estándar)
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Suma de las dos pérdidas
            total_loss = reconstruction_loss + kl_loss

        # Cálculo de los gradientes para la optimización
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Optimización de los pesos utilizando los gradientes calculados
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Actualización de métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Retorno de las métricas actualizadas
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }