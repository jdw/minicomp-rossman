import numpy as np

class LastLazyEstimator:
    def __init__(self, **kwargs):
        self.last_value = {}

    def fit(self, X, y):
        tmp = X.copy()
        tmp["Sales"] = y
        unique_store_id = tmp["Store"].unique()
        for u in unique_store_id:
            self.last_value[u] = tmp[tmp["Store"]==u]["Sales"].values[-1]

    def predict(self,X):
        tmp = X["Store"].apply(lambda x: self.set_last(x))
        return tmp.values

    def set_last(self, store_id):
        if store_id in self.last_value.keys():
            return self.last_value[store_id]
        else:
            return sum(self.last_value.values())/len(self.last_value.values())