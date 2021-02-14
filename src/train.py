from src.utility.load_data import load_data
from metric import metric
from metric import sklearn_metric
from models.mean_lazy_estimator import MeanLazyEstimator
from models.last_lazy_estimator import LastLazyEstimator
from utility.report_name import get_report_name
from src.utility.test_train_split import get_train_test_split

from src.models.default_pipeline import DefaultPipeline

import xgboost as xg

from sklearn.model_selection import ParameterGrid

import argparse
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Trains a model on the rossman dataset')
parser.add_argument('--processed-data',
                    type=str,
                    help='Path to joblib file',
                    default="../data/processed/clean_data.joblib")
parser.add_argument('--output-basename',
                    type=str,
                    help='Default part of filenames in results/',
                    default="test")
parser.add_argument('--final',
                    type=bool,
                    help='If true no test-train-split is performed',
                    default=False)
args = parser.parse_args()

PROCESSED_DATA = args.processed_data
# output_file = open(get_report_name(args.output_basename), "a")
FINAL = args.final

data = load_data("../data/train.csv", PROCESSED_DATA)
trainX, trainY, testX, testY = get_train_test_split(data,
                                                    train_end="2013-07-31",
                                                    test_start="2013-08-01",
                                                    final=FINAL,
                                                    contamination=1e-4)

# lazy = LastLazyEstimator()
# lazy.fit(trainX, trainY)
# print(metric( lazy.predict(trainX), trainY.values ))
# print(metric( lazy.predict(testX), testY.values ))

grid = [{
    # "gamma": [0.001, 0.01, 0.1],
    # "learning_rate": [0.01,0.1,0.2,0.5],
    # "max_depth": [3,4,5,6,7,8,9,10],
    # "n_estimators": [50,100,150, 200],
    "gamma": [0.001],
    "learning_rate": [0.1],
    "max_depth": [12],
    "n_estimators": [26],
    "reg_alpha": [0, 50, 100, 150],
    "reg_lambda": [0.4]
}
]
param_grid = ParameterGrid(grid)

param_result = []
min_val_score = 1e100
best_model = None
best_param = {}

for p in param_grid:
    model = DefaultPipeline(xg.XGBRegressor(**p))
    model.fit(trainX, trainY)

    train_score = metric(model.predict(trainX), trainY.values)
    val_score = metric(model.predict(testX), testY.values)

    print(p, train_score, val_score)
    param_result.append((p, train_score, val_score))

    if val_score < min_val_score:
        min_val_score = val_score
        best_model = model
        best_param = p

joblib.dump(param_result, "param_results_finer.joblib")

if not FINAL:
    training_pmse = f"Training percentage mean squared error {metric(model.predict(trainX), trainY.values)}"
    test_pmse = f"Test percentage mean squared error {metric(model.predict(testX), testY.values)}"

    print(training_pmse)
    print(test_pmse)

    # output_file.write(training_pmse)
    # output_file.write('\n')
    # output_file.write(test_pmse)
    # output_file.close()

joblib.dump(best_model, "../models/model.joblib")
