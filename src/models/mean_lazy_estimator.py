import numpy as np

class MeanLazyEstimator:
    def __init__(self):
        self.mean = {}
        self.global_mean = 0

    def fit(self, X, y):
        tmp = X[["Store"]].copy()
        tmp["Sales"] = y
        self.mean = tmp.groupby(by="Store").mean()["Sales"]
        self.mean.dropna(inplace=True)
        self.global_mean = np.mean(y)

    def predict(self, X):
        tmp = X.copy()
        tmp["mean"] = tmp["Store"].apply(lambda x: self.get_mean(x))
        return tmp["mean"].values


    def get_mean(self, store_id):
        if store_id in self.mean.keys():
            return self.mean.loc[store_id]
        else:
            return self.global_mean