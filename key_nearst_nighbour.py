import numpy as np
import os
os.system("cls")

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None
    

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2-x1)**2))
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_new):
        predictions = []
        for x in x_new:
            ditances = [self.euclidean_distance(x, i) for i in self.x_train]
            k_indexes = np.argsort(ditances)[:self.k]
            k_labels = [self.y_train(i) for i in k_indexes]
            common = np.bincount(k_labels).argmax()
            predictions.append(common)
        return np.array(predictions)
