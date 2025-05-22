# import the ncessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation value of a given input
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # compute the derivative of the sigmoid function assuming 
    # that x has already passed through the sigmoid
    # activation function
    return x * (1 - x)

def predict(X, W):
    # take the dot product between our features and the weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply the step function threshold 
    preds[preds <= 0.5] = 0 # any thing less than or equal 0.5 set to 0
    preds[preds > 0] = 1 # remaining will be 1

    return preds


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, 
    help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, 
    help="learning rate")
args = vars(ap.parse_args())


# generate some data blobs
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, 
    cluster_std=5, random_state=1)
y = y.reshape((y.shape[0], 1)) # make y a column vector


# add a column of 1's in the feature matrix.
# this called the biased trick. Treat the bais 
# as a trainable paramter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]


# partition the data into a training an test split
(trainX, testX, trainY, testY) = train_test_split(X, y, 
    test_size=0.5, random_state=42)


# initialize our weight matrix and losses list
W = np.random.randn(X.shape[1], 1)
losses = []


# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):

    # get the predictions based on the current weights
    preds = sigmoid_activation(trainX.dot(W))

    # compute the error: difference between predictions and true values
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)


    # compute the gradient 
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)

    # update the weights
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] epoch={int(epoch + 1)}, loss={loss:.7f}")
    
# evalute our model
print("[INFO] evaluating ...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# plot the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()