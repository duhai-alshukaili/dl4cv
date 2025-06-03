# import the ncessary packages
import keras
from keras import layers
from keras import backend as K


import tensorflow as tf


class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        model = keras.Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        

        # Input Layer
        model.add(layers.Input(shape=inputShape))

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(layers.Conv2D(32, (3,3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(32, (3,3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(layers.Conv2D(64, (3,3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers. BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(64, (3,3), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers. BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))

        # first and only set of FC => RELU 
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Dropout(0.5))

        # softmax classifier
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))

        return model
