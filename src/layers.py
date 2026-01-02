import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class GELULayer(Layer):
    """Approximate GELU (MATLAB compatible)"""
    def call(self, inputs):
        return 0.5 * inputs * (
            1 + tf.tanh(
                tf.sqrt(2 / np.pi) *
                (inputs + 0.044715 * tf.pow(inputs, 3))
            )
        )


class SwishLayer(Layer):
    """Swish activation with fixed beta"""
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = tf.constant(beta, dtype=tf.float32)

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.beta * inputs)


class FixedPReLULayer(Layer):
    """Non-trainable PReLU"""
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, inputs):
        return tf.maximum(0.0, inputs) + self.alpha * tf.minimum(0.0, inputs)
