from src.models.default_pipeline import DefaultPipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SVMPipe(DefaultPipeline):
    def __init__(self, kwargs):
        self.pass_arguments(kwargs)

    def pass_arguments(self, kwargs):
        super().__init__(SVR(**kwargs))

    def set_pipeline(self, regressor):
        self.pipe = Pipeline([
            ("pre", self.ct),
            ("scl", StandardScaler()),
            ("xgb", regressor)
        ])
