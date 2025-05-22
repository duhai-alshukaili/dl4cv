import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
import argparse

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, 
    help="path to the input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
    help="# of nearest neighnors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of jobs for k-NN distance (-1 uses all avaiable cores)")

args = vars(ap.parse_args())

# get the list of images we will be handling
print("[INFO] loading images ...")
image_paths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the 
# dataset, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
[data, labels] = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072)) # 3072 = 32 * 32 * 3

# show some information on memory consumption of the images
print(f"[INFO] features matrix: {(data.nbytes / (1024 * 1024)):.1f}MB")

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits 75% for
# training and 25% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.25, random_state=42)


# train and evaluate the k-NN classifier
print("[INFO] evaluating k-NN classifier ...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"], metric="manhattan")
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
