import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras import saving
from imutils import paths
import numpy as np
import argparse
import cv2


# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to the input dataset")
ap.add_argument("-m", "--model", required=True, 
    help="path to pre-trained model model")
args = vars(ap.parse_args())

# intitalize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(20,))
imagePaths = imagePaths[idxs]


# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scal the raw pixel
# intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths=imagePaths, verbose=500)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = saving.load_model(args["model"])

# make predication on the images
print("[INFO] predicting...")
preds = model.predict(data,batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):

    image = cv2.imread(imagePath)
    cv2.putText(image, f"Label: {classLabels[preds[i]]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

