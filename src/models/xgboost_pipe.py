from src.models.default_pipeline import DefaultPipeline
import xgboost as xg
import joblib


class XGBoostPipe(DefaultPipeline):
    def __init__(self, kwargs):
        self.pass_arguments(kwargs)

    def pass_arguments(self, kwargs):
        super().__init__(xg.XGBRegressor(**kwargs))


if __name__ == "__main__":
    model = XGBoostPipe({})
    joblib.dump(model, "../../models/base/xgboostpipe.joblib")
