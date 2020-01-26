import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import numpy as np

def create_discriminator():

    tf.keras.backend.set_floatx('float64')

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

    tf.keras.backend.set_floatx('float64')

    model = keras.Sequential(name="generator")

    model.add(layers.Dense(100, input_shape=(100,), activation="relu"))
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
                 discriminator_opt = tf.keras.optimizers.Adam(),
                 generator_opt = tf.keras.optimizers.Adam(),
                 loss_fn = losses.BinaryCrossentropy(from_logits=True),
                 batch_size=32):

        # models
        self.discriminator = discriminator
        self.generator = generator

        # optimizer
        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt

        self.batch_size = batch_size
        self.loss_fn = loss_fn

    def __loss__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def __noise__(self, n_features, size=1):
        return np.array([np.random.uniform(size=n_features)
                         for i in range(size)])

    def generator_predict(self, noise_vector):
        return self.generator(noise_vector)

    def train_discriminator_step(self, X_real, verbose=True):

        # real
        n_real = len(X_real)
        y_real = np.repeat(1.0, n_real)

        # fake
        y_fake = np.repeat(0.0, n_real)
        noise_vectors = self.__noise__(n_features=100, size=n_real)
        X_fake = self.generator(noise_vectors)

        if verbose:
            print("Format X_fake: ", X_fake.shape)
            print("Format X_real: ", X_real.shape)

        # create dataset
        X = np.concatenate([X_real, X_fake], axis=0)
        y = np.concatenate([y_real, y_fake], axis=0)

        train_dataset = tf.data.Dataset.from_tensor_slices((X,y))
        train_dataset = train_dataset.shuffle(2*n_real).batch(self.batch_size)

        # one epoch step
        loss_total = 0.0
        for (X_batch, y_batch) in train_dataset:
            with tf.GradientTape() as g:
                g.watch(self.discriminator.weights)
                y_pred = self.discriminator(X_batch)
                loss_batch = self.__loss__(y_batch, y_pred)
                grads = g.gradient(loss_batch, self.discriminator.weights)
            self.discriminator_opt.apply_gradients(zip(grads,
                                                       self.discriminator.weights))
            loss_total = loss_total + loss_batch.numpy()
        return loss_total

    def fit_discriminator(self, X_real, epochs=10):

        for epoch in range(epochs):
            loss_total = self.train_discriminator_step(X_real)
            print(loss_total)

    def train_generator_step(self, train_dataset):

        loss_total = 0.0
        for (noise_vectors, y_fake) in train_dataset:
            with tf.GradientTape() as g:
                g.watch(self.generator.weights)
                X_fake = self.generator(noise_vectors)
                y_pred = self.discriminator(X_fake)
                loss_batch = self.__loss__(y_fake, y_pred)
                grads = g.gradient(loss_batch, self.generator.weights)
            self.generator_opt.apply_gradients(zip(grads,
                                                    self.generator.weights))
            loss_total = loss_total + loss_batch.numpy()
        return loss_total

    def fit_generator(self, n_size=100, epochs=10):

                # create fake images fake
        y_fake = np.repeat(1.0, n_size)
        noise_vectors = self.__noise__(n_features=100, size=n_size)

        # create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((noise_vectors,y_fake))
        train_dataset = train_dataset.batch(self.batch_size)


        for epoch in range(epochs):
            loss_total = self.train_generator_step(train_dataset)
            print(loss_total)
            #print(self.generator.weights)


# init model
gan = GAN(discriminator=discriminator,
    generator=generator, generator_opt=tf.keras.optimizers.Adam(learning_rate=0.1))

# test generator
noise_vector = gan.__noise__(100)
y_pred = gan.generator_predict(noise_vector)
y_pred.shape

noise_vectors = gan.__noise__(100, 10)
y_preds = gan.generator_predict(noise_vectors)
y_preds.shape

# load mnist data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_s = np.expand_dims(X_train, -1) / 255.
X_test_s = np.expand_dims(X_test, -1) / 255.

# test discriminator train method
gan.train_discriminator_step(X_real=X_test_s[:100])
gan.fit_discriminator(X_real=X_test_s[:100])

# test
gan.discriminator(X_train_s[:10])
gan.discriminator(y_preds[:10])

# test train generator
gan.fit_generator(n_size=100)

noise_vectors = gan.__noise__(100, 10)
y_preds = gan.generator_predict(noise_vectors)

gan.discriminator(y_preds)



