# Esta capa de muestreo (Sampling) es la capa de cuello de botella del VAE
# Utiliza las salidas de dos capas densas, z_mean y z_log_var, como entrada,
# las convierte en una distribución normal y las pasa a la capa decodificadora

import tensorflow as tf
import keras
from keras import layers


class Sampling(layers.Layer):
    """
    Utiliza (mean, log_var) para muestrear z (vector que codifica un dígito).
    Args:
        mean: media de la distribución.
        log_var: logaritmo de la varianza
    """
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon