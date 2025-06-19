import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.callbacks import TrainingMonitor
from keras import optimizers
from keras import datasets
from keras import callbacks

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Path to the output directory")
ap.add_argument("-e", "--epochs", type=int, default=10,
                help="Number of training epochs (default: 10)")
args = vars(ap.parse_args())


# show information on the process ID
print(f"[INFO] process ID: {os.getpid()}")


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

# construct and set the callback
fig_path = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
json_path = os.path.sep.join([args["output"], f"{os.getpid()}.json"])
callbacks = [TrainingMonitor(figPath=fig_path, jsonPath=json_path)]

# initialize the optimizer and model
print("[INFO] compiling model ...")
opt = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
    metrics=["accuracy"])

# get the number of epochs
epochs = args["epochs"]

# train the network
print(f"[INFO] training network for {epochs} epochs ...")
model.fit(trainX, trainY, validation_data=(testX, testY), 
    batch_size=64, callbacks=callbacks, epochs=epochs, verbose=1)