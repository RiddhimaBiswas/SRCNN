import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss
from tensorflow.nn import conv2d


class CustomLoss(Loss):
    def __init__(self, alpha, beta, lambda_edge, lambda_noise, lambda_entropy):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_edge = lambda_edge
        self.lambda_noise = lambda_noise
        self.lambda_entropy = lambda_entropy

    def call(self, y_true, y_pred):
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        nrmse = self._nrmse(y_true, y_pred)
        msssim = self._msssim(y_true, y_pred)
        edge = self._edge_strength(y_pred)
        noise = self._noise_level(y_pred)
        entropy = self._entropy(y_pred)

        return (
            self.alpha * nrmse +
            self.beta * msssim -
            self.lambda_edge * edge +
            self.lambda_noise * noise -
            self.lambda_entropy * entropy
        )

    def _nrmse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.sqrt(mse) / (tf.reduce_max(y_true) - tf.reduce_min(y_true) + 1e-6)

    def _msssim(self, y_true, y_pred):
        return 1.0 - tf.reduce_mean(
            tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
        )

    def _edge_strength(self, img):
        sobel = tf.image.sobel_edges(img)
        grad = tf.sqrt(tf.square(sobel[..., 0]) + tf.square(sobel[..., 1]))
        return tf.reduce_mean(grad)

    def _noise_level(self, img):
        kernel = tf.ones((3, 3, 1, 1)) / 9.0
        mean = conv2d(img, kernel, 1, 'SAME')
        var = conv2d(tf.square(img), kernel, 1, 'SAME') - tf.square(mean)
        return tf.reduce_mean(tf.abs(var))

    def _entropy(self, img):
        bins = 256
        centers = tf.linspace(0.0, 1.0, bins)
        diff = tf.expand_dims(img, -1) - centers
        hist = tf.reduce_sum(tf.exp(-diff ** 2 / 0.01), axis=[1, 2])
        hist /= tf.reduce_sum(hist, axis=-1, keepdims=True)
        return -tf.reduce_mean(hist * tf.math.log(hist + 1e-8))
