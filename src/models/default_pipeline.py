import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class DefaultPipeline():
    def __init__(self, regressor):
        self.set_preprocessing()
        self.set_pipeline(regressor)

    def set_preprocessing(self):
        self.ct = ColumnTransformer([
            ("target_enc",
             ce.target_encoder.TargetEncoder(cols=["Store", 'StoreType', 'Assortment', 'PromoInterval']),
             ["Store", 'StoreType', 'Assortment', 'PromoInterval']),
            ("one_hot_enc",
             OneHotEncoder(handle_unknown="ignore"),
             ['StateHoliday'])
        ], remainder='passthrough')

    def set_pipeline(self, regressor):
        self.pipe = Pipeline([
            ("pre", self.ct),
            ("xgb", regressor)
        ])

    def fit(self, X, y):
        self.pipe.fit(X,y)

    def predict(self, X):
        return self.pipe.predict(X)
