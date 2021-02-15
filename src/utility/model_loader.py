from src.models.xgboost_pipe import XGBoostPipe
from src.models.last_lazy_estimator import LastLazyEstimator
from src.models.mean_lazy_estimator import MeanLazyEstimator
from src.models.random_forest_pipe import RandomForestPipe
from src.models.lassopipe import LassoPipe
from src.models.svmpipe import SVMPipe
from src.models.extratreepipe import ExtraTreePipe

class ModelLoader():
    def __init__(self):
        self.models = {
            "randomforestpipe": RandomForestPipe({}),
            "xgboostpipe": XGBoostPipe({}),
            "extratreepipe": ExtraTreePipe({}),
            "lassopipe": LassoPipe({}),
            "svmpipe": SVMPipe({}),
            "lastlazy": LastLazyEstimator(),
            "meanlazy": MeanLazyEstimator()
        }

    def get_model(self, model="meanlazy", parameters={}):
        if model in ["lastlazy", "meanlazy"]:
            return self.models[model]
        else:
            self.models[model].pass_arguments(parameters)
            return self.models[model]
