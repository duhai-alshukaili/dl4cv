import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import ShallowNet
from keras import optimizers
from keras import datasets

import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=10,
                help="Number of training epochs (default: 10)")
args = vars(ap.parse_args())

# grab the mnist data set.
print("[INFO] accessing CIFAR-10 ...")
((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()

# scale data to range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX  = testX.astype("float32") / 255.0

# convert labes from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY) 

# initialize the label names for the CIFAR-10 datasets
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"]



# initialize the optimizer and model
print("[INFO] compiling model ...")
opt = optimizers.SGD(learning_rate=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
    metrics=["accuracy"])

# get the number of epochs
epochs = args["epochs"]

# train the network
print(f"[INFO] training network for {epochs} epochs ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
    batch_size=32, epochs=epochs, verbose=1)

print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), 
    predictions.argmax(axis=1), 
    target_names=labelNames))


# plot the training loss and accuracy
plt.style.use("fivethirtyeight")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("shallownet_cifar10.png")
