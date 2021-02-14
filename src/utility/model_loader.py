from src.models.xgboost_pipe import XGBoostPipe
from src.models.last_lazy_estimator import LastLazyEstimator
from src.models.mean_lazy_estimator import MeanLazyEstimator


class ModelLoader():
    def __init__(self):
        self.models = {
            "xgboostpipe": XGBoostPipe({}),
            "lastlazy": LastLazyEstimator(),
            "meanlazy": MeanLazyEstimator()
        }

    def get_model(self, model="meanlazy", parameters={}):
        try:
            self.models[model].pass_arguments(parameters)
            return self.models[model]
        except AttributeError:
            return self.models[model]
