from src.models.default_pipeline import DefaultPipeline
from sklearn.ensemble import RandomForestRegressor

class RandomForestPipe(DefaultPipeline):
    def __init__(self, kwargs):
        self.pass_arguments(kwargs)

    def pass_arguments(self, kwargs):
        super().__init__(RandomForestRegressor(**kwargs))
