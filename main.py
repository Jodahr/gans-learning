#!/usr/bin/env python

# standard libs
import pandas as pd
import subprocess
import numpy as np
import functools
import pickle
from matplotlib import pyplot as plt

# ML and keras
import keras
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications, Model, metrics
from keras.callbacks import History, TensorBoard, EarlyStopping
from keras.layers.convolutional import Convolution2D, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

# custom module (old)
import utils as ut

# simple custom generator
train_data_path = '/home/jodahr/Software/Development/DogBreeds/data/train'
data_gen = ut.predictGenerator(train_data_path, target_size=(128, 128))

# test generator
plt.imshow(next(data_gen)[0])

# randomDim
random_dim = 100


def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Convolution2D(32, (3, 3), input_shape=(3, 128, 128)))
    discriminator.add(Activation('relu'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))

    discriminator.add(Conv2D(32, (3, 3)))
    discriminator.add(Activation('relu'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))

    discriminator.add(Conv2D(64, (3, 3)))
    discriminator.add(Activation('relu'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))

    # this converts our 3D feature maps to 1D feature vectors
    discriminator.add(Flatten())
    discriminator.add(Dense(64))
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    discriminator.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

    return discriminator

# generator vanilla network


def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(8*8*1024, input_dim=random_dim,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))

    generator.add(keras.layers.Reshape((8, 8, 1024)))
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(512, (5, 5), strides=[
                  2, 2], padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(256, (5, 5), strides=[
                  2, 2], padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(128, (5, 5), strides=[
                  2, 2], padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(64, (5, 5), strides=[
                  2, 2], padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(3, (5, 5), strides=[
                  1, 1], padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Activation('tanh'))

    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator


gan_input = keras.engine.input_layer.Input(batch_shape=(random_dim,))
gan_input
gen = get_generator(optimizer='adam')
gen.predict([[np.zeros(100)]])[0]
tf.__version__
