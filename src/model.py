import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .layers import *
from .loss import *
from .filters import *


def create_cnn(position, input_size, batch_size, epochs,
               image_dir, target_dir, analyze=False):

    inputs = Input(shape=(None, None, 1))
    x = inputs

    upscaled = layers.Lambda(
        lambda i: tf.image.resize(i,
            [tf.shape(i)[1]*2, tf.shape(i)[2]*2],
            method='bicubic')
    )(inputs)

    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = SwishLayer(1.0)(x)

    x = layers.Conv2DTranspose(1, 3, strides=2, padding='same')(x)
    x = GELULayer()(x)

    x = layers.Add()([x, upscaled])
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    loss_fn = CustomLoss(1, 1, 0.05, 0.05, 0.05)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9),
        loss=loss_fn
    )

    gen = ImageDataGenerator(rescale=1./255)
    input_gen = gen.flow_from_directory(
        image_dir,
        target_size=(input_size, input_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None
    )

    target_gen = gen.flow_from_directory(
        target_dir,
        target_size=(input_size*2, input_size*2),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None
    )

    def pair():
        while True:
            yield next(input_gen), next(target_gen)

    history = model.fit(
        pair(),
        steps_per_epoch=math.ceil(input_gen.n / batch_size),
        epochs=epochs,
        verbose=2 if analyze else 0
    )

    if analyze:
        plt.plot(history.history['loss'])
        plt.show()

    return history.history['loss'][-1]
