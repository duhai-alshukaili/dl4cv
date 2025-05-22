# import the ncessary packages
import tensorflow as tf

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the data format
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        # apply the keras utility function that 
        # correctly rearranges the dimensions of the image
        return tf.keras.utils.img_to_array(image, data_format=self.dataFormat)
        