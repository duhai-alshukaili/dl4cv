import keras
from keras import layers
from keras import backend as K

class LeNet:

    @staticmethod
    def build(width, height, depth, classes) -> keras.Sequential:

        #intialize the model
        model = keras.Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # first set of CONV => RELU => POOL layers
        model.add(layers.Input(shape=inputShape))
        model.add(layers.Conv2D(20, (5, 5), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # second set of CONV => RELU => POOL layers
        model.add(layers.Conv2D(50, (5, 5), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # first and only set of FC => RELU
        model.add(layers.Flatten())
        model.add(layers.Dense(50))
        model.add(layers.Activation("relu"))

        # softmax classifier
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))

        # return the constructed network architecture
        return model