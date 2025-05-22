import numpy as np

class Perceptron:

    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply the setp function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # add a column of 1's as the last entry in the fatures
        # matrix - this is called the bias trick. Thus the bias becomes
        # as a trainable paramter.
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for _ in np.arange(0, epochs):

            # loop over each iniviudal example
            for (x, target) in zip(X, y):

                # take the dot product between the features
                # and the weight vector then apply the step function
                p = self.step(np.dot(x, self.W))

                # perform the weight update only
                # if the prediction does not match
                # the target
                if p != target:

                    error = p - target

                    self.W += -self.alpha * error * x

    def predict(self, X, add_bias=True):

        # ensure that our input is a matrix
        X = np.atleast_2d(X)

        if add_bias:
            X = np.c_[X, np.ones((X.shape[0]))]


        return self.step(np.dot(X, self.W))

