import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from keras import optimizers
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to the input dataset")
ap.add_argument("-m", "--model", required=True, 
    help="path to output model")
ap.add_argument("-e", "--epochs", type=int, default=10,
                help="Number of training epochs (default: 10)")
args = vars(ap.parse_args())

# get the list of images we will be handling
print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scal the raw pixel
# intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths=imagePaths, verbose=500)
data = data.astype("float") / 255.0

# split the data into training and testing data
# use 75% for training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.25, random_state=42)


# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model ...")
opt = optimizers.SGD(learning_rate=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
    metrics=["accuracy"])

# get the number of epochs
epochs = args["epochs"]

# train the network
print(f"[INFO] training network for {epochs} epochs ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
    batch_size=32, epochs=epochs, verbose=1)

# save the model to the disk
print("[INFO] serializing network ...")
model.save(args["model"])

print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), 
    predictions.argmax(axis=1), 
    target_names=["cat", "dog", "panda"]))


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
plt.savefig("shallownet_animals.png")