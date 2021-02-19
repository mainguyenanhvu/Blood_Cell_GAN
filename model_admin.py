import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RNN, LSTM, GRU, SimpleRNN
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU # comment dong nay neu may ban khong ho tro CuDNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Dropout,  BatchNormalization, ELU
from tensorflow.keras.layers import TimeDistributed, Embedding, Bidirectional
from tensorflow.keras.layers import Flatten, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D

INPUT_SHAPE = None
INPUT_DIM = None
CLASS_NUM = None
NOISE_SHAPE = None

def naive_GAN():
    generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(150 * 150 * 3, input_shape=[NOISE_SHAPE,]),
    tf.keras.layers.Reshape(INPUT_SHAPE),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                    activation="selu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(3,kernel_size=3, strides=3, padding="valid",
                                    activation="selu"),
    tf.keras.layers.MaxPool2D((6,6))
    ])


    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                            activation=tf.keras.layers.LeakyReLU(0.2),
                            input_shape=INPUT_SHAPE),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                            activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    gan = tf.keras.models.Sequential([generator, discriminator])
    return gan, generator, discriminator


def create_model(model_name, dataset):
    global INPUT_SHAPE
    global CLASS_NUM
    global NOISE_SHAPE
    CLASS_NUM = dataset.classnum
    datasetshape = dataset.shape
    NOISE_SHAPE = dataset.noise_shape
    if len(datasetshape) > 0:
        INPUT_SHAPE = datasetshape
    else:
        INPUT_SHAPE = (datasetshape[0],)
    dic_model = {'naive_GAN': naive_GAN}
    if model_name in dic_model.keys():
        return dic_model[model_name]()
    else:
        return dic_model['naive_GAN']()