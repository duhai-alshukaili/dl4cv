import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras import optimizers
from keras import callbacks
from keras import datasets

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="Path to the weights directory")
ap.add_argument("-e", "--epochs", type=int, default=40,
                help="Number of training epochs (default: 40)")
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
testY = lb.transform(testY) 

# initialize the label names for the CIFAR-10 datasets
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"]

# get the number of epochs
epochs = args["epochs"]

# initialize the optimizer and model
print("[INFO] compiling model ...")
opt = optimizers.SGD(learning_rate=0.01, 
                     weight_decay=0.01/epochs, 
                     momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
    metrics=["accuracy"])

# construct 
fname = os.path.sep.join([args["weights"], 
                         "weights-{epoch:03d}-{val_loss:.4f}.hdf"])

checkpoint = callbacks.ModelCheckpoint(fname, monitor="val_loss", mode="min", 
                                       save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print(f"[INFO] training network for {epochs} epochs ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
    batch_size=64, epochs=epochs, verbose=2, callbacks=callbacks)