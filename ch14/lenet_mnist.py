import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from pyimagesearch.nn.conv import LeNet
from keras import optimizers
from keras import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse


# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=10,
                help="Number of training epochs (default: 10)")
args = vars(ap.parse_args())

# grab the MNIST dataset. It is about 11MB of download
((trainData, trainLabels), (testData, testLabels)) = datasets.mnist.load_data()

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))

# otherwise, we are using "chnnels_last" and out design matrix is:
# num_samples x rows x columns x depth
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale the data on the range [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# conver the labes from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels  = le.transform(testLabels)

# initialize the optimizer and model
print("[INFO] compiling the model")
opt = optimizers.SGD(learning_rate=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# get the number of epochs
epochs = args["epochs"]

# train the network
H = model.fit(trainData, trainLabels, 
              validation_data=(testData, testLabels), 
              batch_size=128, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testData, batch_size=128)
print(classification_report(testLabels.argmax(axis=1), 
      predictions.argmax(axis=1), 
      target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("lenet_mnist.png")
