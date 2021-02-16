import fbprophet
import pandas as pd


class Prophet:
    def __init__(self):
        self.data = {}
        self.store_id = []

    def fit(self, X, y):
        self.store_id = X["Store"].unique()
        for s in self.store_id:
            map_s = (X["Store"]==s)
            df = X[map_s].copy()
            df["y"] = y[map_s]
            df["ds"] = X[map_s].index
            self.data[s] = df[["ds", "y"]]

    def predict(self, X):
        df = X.copy()
        df["predict"] = 0
        df["ds"] = df.index
        for s in self.store_id:
            model = fbprophet.Prophet(yearly_seasonality=True, daily_seasonality=True)
            model.fit( self.data[s] )
            tmp = model.predict(df[df["Store"]==s])
            df.loc[df["Store"]==s,"predict"] = tmp["yhat"].values

        return df["predict"].values