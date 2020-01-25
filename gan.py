import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

def create_discriminator():

    model = keras.Sequential(name="discriminator")

    # convs
    model.add(layers.Conv2D(filters=32, kernel_size=(4,4),
              activation="relu", input_shape=(28,28,1)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(2,2),
              activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # flatten
    model.add(layers.Flatten())

    # classification
    model.add(layers.Dense(10, activation="relu"))

    # final dense layer
    model.add(layers.Dense(units=1,activation="sigmoid"))

    return model


def create_generator():

    model = keras.Sequential(name="generator")

    model.add(layers.Dense(10, input_shape=(100,), activation="relu"))
    model.add(layers.Reshape(target_shape=(10,10,1)))

    model.add(layers.Conv2DTranspose(filters=5,
                                      kernel_size=(3,3), strides=(2,2)))

    model.add(layers.Conv2DTranspose(filters=5,
                                      kernel_size=(3,3), strides=(1,1)))

    model.add(layers.Conv2DTranspose(filters=1,
                                      kernel_size=(6,6), strides=(1,1)))

    return model


generator = create_generator()
generator.summary()

discriminator = create_discriminator()
discriminator.summary()

class GAN():

    def __init__(self, discriminator, generator,
                 discriminator_opt = keras.optimizers.Adam(),
                 generator_opt = keras.optimizers.Adam(),
                 batch_size=32):

        # models
        self.discriminator = discriminator
        self.generator = generator

        # optimizer
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt

        self.batch_size = batch_size


GAN(discriminator=discriminator,
    generator=generator)
