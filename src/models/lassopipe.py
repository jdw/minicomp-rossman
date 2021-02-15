from src.models.default_pipeline import DefaultPipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class LassoPipe(DefaultPipeline):
    def __init__(self, kwargs):
        self.pass_arguments(kwargs)

    def pass_arguments(self, kwargs):
        super().__init__(Lasso(**kwargs))

    def set_pipeline(self, regressor):
        self.pipe = Pipeline([
            ("pre", self.ct),
            ("scl", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("xgb", regressor)
        ])