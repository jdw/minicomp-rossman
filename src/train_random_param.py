from src.metric import metric
from src.utility.model_loader import ModelLoader
from src.utility.load_data import load_data
from src.utility.test_train_split import get_train_test_split
from sklearn.model_selection import ParameterGrid

import argparse
import json
import joblib
import tqdm
import random

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class RandomParameterSearch:
    def __init__(self, path_to_instructions):
        with open(path_to_instructions) as json_file:
            self.instructions = json.load(json_file)
        self.trainX, self.trainY, self.testX, self.testY = self.get_data(self.instructions["train"])

    def get_data(self, path_to_csv):
        data = load_data(path_to_csv)
        return get_train_test_split(data,
                                    train_end="2013-07-31",
                                    test_start="2013-08-01",
                                    final=False,
                                    contamination=1e-4)

    def reset(self):
        self.param_grid = ParameterGrid([self.instructions["grid"]])
        self.param_result = []
        self.min_val_score = 1e100
        self.best_model = None
        self.best_param = {}

    def search(self, max_runs=100):
        self.reset()
        self.model_loader = ModelLoader()
        pbar = self.get_loop(max_runs)
        pbar.set_description(f"Best Validation Score: {0.:10.4f}")
        for p in pbar:
            self.train(p)
            pbar.set_description(f"Best Validation Score: {self.min_val_score:10.4f}")

    def get_loop(self, max_runs):
        if max_runs >= len(self.param_grid):
            return tqdm.tqdm(self.param_grid)
        else:
            return tqdm.tqdm(random.choices(self.param_grid, k=max_runs))

    def train(self, parameter):
        model = self.model_loader.get_model(model=self.instructions["model"], parameters=parameter)
        model.fit(self.trainX, self.trainY)

        train_score = metric(model.predict(self.trainX), self.trainY.values)
        val_score = metric(model.predict(self.testX), self.testY.values)

        self.param_result.append(
            {
                "parameters": parameter,
                "train_score": train_score,
                "val_score": val_score
            }
        )

        if val_score < self.min_val_score:
            self.min_val_score = val_score
            self.best_model = model
            self.best_param = parameter
            joblib.dump(model, self.instructions["model_output"])

    def write_report(self):
        report = {
            "model": self.instructions["model"],
            "best_model": self.instructions["model_output"],
            "best_parameter": self.best_param,
            "val_score": self.min_val_score,
            "parameter_grid": self.instructions["grid"],
            "results": self.param_result
        }

        with open(self.instructions["report_output"], 'w') as outfile:
            json.dump(report, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Searches for best parameter set from random choices, according to instructions and reports to output file.')
    parser.add_argument('--instructions',
                        type=str,
                        help='Path to instructions file (json)',
                        default="instructions/extratreepipe_wide.json")
    parser.add_argument('--max_runs',
                        type=int,
                        help='Number of random choices',
                        default=200)
    args = parser.parse_args()

    rnd = RandomParameterSearch(args.instructions)
    rnd.search(max_runs=args.max_runs)
    rnd.write_report()
