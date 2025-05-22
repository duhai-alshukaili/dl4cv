import numpy as np 
import cv2
import os 

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):

        # store the image preprocessors
        self.preprocessors = preprocessors

        # no preprocessors are given
        if self.preprocessors is None:
            self.preprocessors = [] 
        
    def load(self, image_paths, verbose=-1):

        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            # load the image and extract the class label assuming
            # that out path has the following format
            # /path/to/dataset/{class}/{image}.jpeg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            # preprocess the image if not we have some preprocessots
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i+1}/{len(image_paths)}")
        

        # return a tuple of data and labels
        return (np.array(data), np.array(labels))