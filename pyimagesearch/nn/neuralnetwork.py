import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the lise of weight matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # loop from the index of the first layer 
        # but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weigh matrix coneecting the 
            # number of nodes in each respective layer together
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i])) # normilze the variance of weights

        
        # the last two layers are  speical case. the input 
        # connecion need a bias term but the output does not.
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        # compute and return the sigmoid activation value 
        # for a give input value
        return 1.0 /  (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING
        # that `x` already has been passed through the `sigmoid` function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # the bias trick
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over indivdual data points and 
            # train our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            
            # check to see if we should display the training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))
    
    def fit_partial(self, x, y):

        # Construct the list of activations. The first is just 
        # the input values which are the input feature vector 
        # itself.
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # loop over the layer in the netwrok
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and 
            # the weight matrix -- this is called the net input
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # compute the net output which is simply applying the
            # nonlinear activation function on the net input
            out = self.sigmoid(net)

            # append the output to the activation list
            A.append(out)
        
        # BACKPROPAGATION
        # the first step in the backpropogation is to compute the 
        # differnce between our predication and the true target
        error = A[-1] - y 

        # from here we need to apply the chain rule to build our
        # list of deltas `D`; the first is the error of the out 
        # multipled in the derivative of our activation function
        D = [error * self.sigmoid_deriv(A[-1])]

        # loop over the layer in reverse to apply the chain rule
        # the last two ignored because we have already taken them
        # into account
        for layer in np.arange(len(A) - 2, 0, -1):

            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        
        # since we looped over the layers in the reverse order
        # we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):

            # update the weight by taking the dot product of the 
            # activations and thier deltas then subtacing these 
            # from the initial weights -- this is where the actual 
            # learning happens
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
    
    def predict(self, X, addBias=True):

        # initialize the ouput predication as the input features --
        # this value will be (forward)  propagated through the network 
        # to obtain the final predication
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p
    
    def calculate_loss(self, X, targets):

        # make predictions for the input data then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss



    



