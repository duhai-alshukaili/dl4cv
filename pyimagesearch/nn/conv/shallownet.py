import keras
from keras import layers
from keras import backend as K


class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes) -> keras.Sequential:

        # the default input shape will "channels last"
        model = keras.Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, width, height)
        
        # define the first (and only) CONV => RELU layer
        model.add(layers.Input(shape=inputShape))
        model.add(layers.Conv2D(32, (3,3), padding="same"))
        model.add(layers.Activation("relu"))

        # softmax classifier
        model.add(layers.Flatten())
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))

        return model

