from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the mnist data set.
print("[INFO] accessing CIFAR-10 ...")
((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()

# scale data to range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX  = testX.astype("float32") / 255.0

# each image in the CIFAR-10 data set is represnted by a 32x32x3
# image, but we need to flatten the image to apply a standard NeuralNetwork
trainX = trainX.reshape((trainX.shape[0], 32 * 32 * 3))
testX  = testX.reshape((testX.shape[0], 32 * 32 * 3))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.transform(testY)

labelNames = ["aiplane", "automobile", "bird", "cat", 
    "deer", "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model = keras.Sequential(
    [
        layers.Dense(1024, input_shape=(3072, ), activation="relu", name="Layer1"),
        layers.Dense(512, activation="relu", name="Layer2"),
        layers.Dense(10, activation="softmax", name="Layer3")
    ]
)

# train the model using SGD
print("[INFO] training the network ...")
sgd = optimizers.SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, 
    metrics=["accuracy"])

# epochs
e = 9
# fit the model and return dictornary the helps plot
# the loss/accuracy plot
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
    epochs=e, batch_size=32)

print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), 
    predictions.argmax(axis=1), 
    target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, e), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, e), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, e), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, e), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])