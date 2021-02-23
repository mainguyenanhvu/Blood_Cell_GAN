import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RNN, LSTM, GRU, SimpleRNN
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU # comment dong nay neu may ban khong ho tro CuDNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Dropout,  BatchNormalization, ELU, LeakyReLU, Concatenate, Multiply
from tensorflow.keras.layers import TimeDistributed, Embedding, Bidirectional
from tensorflow.keras.layers import Flatten, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Reshape, Conv2DTranspose

INPUT_SHAPE = None
INPUT_DIM = None
CLASS_NUM = None
NOISE_SHAPE = None

def naive_GAN():
    flat_input_shape = 1
    for i in INPUT_SHAPE:
        flat_input_shape = flat_input_shape*i
    generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(flat_input_shape, input_shape=[NOISE_SHAPE,]),
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

def vanilla_GAN():
    flat_input_shape = 1
    for i in INPUT_SHAPE:
        flat_input_shape = flat_input_shape*i
    generator = tf.keras.models.Sequential([
        tf.keras.layers.Dense(65536, input_shape=[NOISE_SHAPE,]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((8,8,1024)),
        tf.keras.layers.Conv2DTranspose(512, kernel_size=(5,5), strides=2, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(256, kernel_size=(5,5), strides=2, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(128, kernel_size=(5,5), strides=2, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=(5,5), strides=2, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(3, kernel_size=(5,5), strides=2, padding="SAME")
    ])

    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=2, padding="same", input_shape=INPUT_SHAPE),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Conv2D(128, kernel_size=(5,5), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Conv2D(512, kernel_size=(5,5), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Conv2D(1024, kernel_size=(5,5), strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    gan = tf.keras.models.Sequential([generator, discriminator])
    return gan, generator, discriminator

def AC_GAN():
    flat_input_shape = 1
    for i in INPUT_SHAPE:
        flat_input_shape = flat_input_shape*i

    in_label = Input(shape=(1,))
    li = Embedding(4, 100)(in_label)
    li = Reshape((100,))(li)

    in_lat = Input(shape=(NOISE_SHAPE,))
    merge = Multiply()([in_lat, li])
    merge = Dense(flat_input_shape)(merge)
    out_layer = Reshape(INPUT_SHAPE)(merge)
    #out_layer = Sequential([Reshape(INPUT_SHAPE)])(Dense(flat_input_shape)(merge))
    generator = Model([in_lat, in_label], out_layer)

    inp = Input(shape=INPUT_SHAPE)
    lay = Sequential([Flatten()])(inp)
    out_layer1 = Sequential([Dense(1, activation='sigmoid')])(lay)

    lay = Dense(NOISE_SHAPE)(lay)
    lay = Dense(NOISE_SHAPE)(lay)
    lay = Dense(NOISE_SHAPE)(lay)
    out_layer = Dense(CLASS_NUM, activation='softmax')(lay)
    discriminator = Model(inp, [out_layer,out_layer1])
    gan = None#tf.keras.models.Sequential([generator, discriminator])

    return gan, generator, discriminator


def DCGAN():
    x_shape = 16
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(CLASS_NUM, NOISE_SHAPE)(in_label)
    # linear multiplication
    n_nodes = x_shape * x_shape
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((x_shape, x_shape, 1))(li)
    # image generator input
    in_lat = Input(shape=(NOISE_SHAPE,))

    n_nodes = x_shape * x_shape * 1
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((x_shape, x_shape, 1))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])

    gen = Conv2DTranspose(4, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.4)(gen)
    gen = Conv2DTranspose(8, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.4)(gen)
    gen = Conv2DTranspose(16, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.4)(gen)
    gen = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.4)(gen)
    # output
    out_layer = Conv2D(3, (7,7), activation='tanh', padding='same')(gen)
    # define model
    generator = Model([in_lat, in_label], out_layer)

    inp = Input(shape=INPUT_SHAPE)
    fe = Conv2D(4, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(8, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(16, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(64, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(16, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(8, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer1 = Dense(1, activation='sigmoid')(fe)
    #out_layer1 = Sequential([Dense(1)])(lay)

    fe = Conv2D(4, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(8, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(16, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(inp)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(64, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(32, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(16, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Conv2D(8, (3,3), strides=(2,2), padding='valid')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    out_layer = Dense(CLASS_NUM, activation='softmax')(fe)
    discriminator = Model(inp, [out_layer,out_layer1])
    gan = None#tf.keras.models.Sequential([generator, discriminator])

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
    dic_model = {'naive_GAN': naive_GAN, 'vanilla_GAN': vanilla_GAN, 'AC_GAN': AC_GAN, 'DCGAN': DCGAN}
    if model_name in dic_model.keys():
        return dic_model[model_name]()
    else:
        return dic_model['vanilla_GAN']()