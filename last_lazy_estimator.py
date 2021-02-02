import numpy as np

class LastLazyEstimator:
    def __init__(self):
        self.last_value = 0

    def fit(self, X, y):
        self.last_value = y[-1]

    def predict(self,X):
        return self.last_value * np.ones((len(X),))