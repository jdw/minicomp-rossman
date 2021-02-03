import numpy as np

class MeanLazyEstimator:
    def __init__(self):
        self.mean = 0

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        return self.mean * np.ones( (len(X), ) )