import sys, pathlib

# Append the parent directory of ch07 to sys.path once
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to the input dataset")
args = vars(ap.parse_args())

# grab the list of images paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the images
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the data into training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.25, random_state=42)


# loop over the list of regularizers
for r in [None, "l1", "l2"]:

    # train a SGD classifier using a softmax loss function
    # and specified regularizer for 100 epochs

    print(f"\n[INFO] training model with `{r}` penalty")
    model = SGDClassifier(loss="hinge", penalty=r, max_iter=1000,
        learning_rate="constant", tol=1e-3, eta0=0.01, random_state=12)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print(f"[INFO] `{r} penalty accuracy: {acc * 100:.2f}%`")