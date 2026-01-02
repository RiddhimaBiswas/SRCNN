import numpy as np
import cv2
import tensorflow as tf
from scipy.signal import convolve2d
from keras.initializers import Initializer
from keras.saving import register_keras_serializable


def generate_lap(n):
    q = (n + 1) // 2
    M = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            M[i, j] = min(i + 1, j + 1)

    M = np.block([[M, np.fliplr(M)],
                  [np.flipud(M), np.flipud(np.fliplr(M))]])

    if n % 2 == 1:
        M = np.delete(M, q - 1, 0)
        M = np.delete(M, q - 1, 1)
        M[q - 1, q - 1] = -np.sum(M)
    return M.astype(np.float32)


def sobel_prewitt_filter(n):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    return kernel if np.random.rand() < 0.5 else kernel.T


def create_gabor_filter(n, lambd, theta, sigma, gamma):
    grid = np.arange(-n // 2 + 1, n // 2 + 1)
    x, y = np.meshgrid(grid, grid)
    x_t = x * np.cos(theta) + y * np.sin(theta)
    y_t = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(x_t**2 + gamma**2 * y_t**2) / (2 * sigma**2))
    return (g * np.cos(2 * np.pi * x_t / lambd)).astype(np.float32)


def create_scharr_filter():
    k = np.array([[-3, 0, 3],
                  [-10, 0, 10],
                  [-3, 0, 3]], dtype=np.float32)
    return k if np.random.rand() < 0.5 else k.T


def fspecial_log(n, sigma):
    ax = np.arange(-n // 2 + 1, n // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    h *= (xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)
    return h.astype(np.float32)


def filterselection(t, s, n, ni):
    if t == 3:
        return tf.keras.initializers.HeNormal()
    if t == 4:
        return tf.keras.initializers.GlorotNormal()

    filters = []
    for _ in range(n):
        if t == 6:
            kernel = generate_lap(s)
        elif t == 7:
            kernel = create_gabor_filter(s, 4, np.pi/4, s/6, 0.5)
        elif t == 8:
            kernel = sobel_prewitt_filter(s)
        else:
            kernel = create_scharr_filter()

        filters.append(np.repeat(kernel[:, :, np.newaxis], ni, axis=2))

    return np.stack(filters, axis=3)


@register_keras_serializable()
class CustomInitializer(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None):
        return tf.constant(self.value, dtype=tf.float32)
