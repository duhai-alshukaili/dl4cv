# import the ncessary packages
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(callbacks.Callback):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath=jsonPath
        self.startAt = startAt
    
    def on_train_begin(self, logs={}):

        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # cehck to see if the a starting epoch was supplied
                if self.startAt > 0:
                    # loop over entries in the history lod
                    # and trim entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    
    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuary, etc. 
        # for the entire traing process.
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l
        
        # check to see if the training history should be serialized 
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        # ensure that two epochs has passed before plotting
        if len(self.H["loss"]) > 1:

            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("fivethirtyeight")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()