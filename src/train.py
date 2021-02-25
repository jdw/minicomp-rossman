from src.utility.load_data import load_data
from src.utility.model_loader import ModelLoader
from src.utility.test_train_split import get_train_test_split

from src.metric import metric

import argparse
import json
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Trains a model on the rossman dataset')
parser.add_argument('--processed-data',
                    type=str,
                    help='Path to joblib file',
                    default="../data/processed/clean_data.joblib")
parser.add_argument('--report',
                    type=str,
                    help='Report to read best parameters from',
                    default="../results/prophet.json")
parser.add_argument('--final',
                    type=bool,
                    help='If true no test-train-split is performed',
                    default=True)
args = parser.parse_args()

PROCESSED_DATA = args.processed_data
REPORT = args.report
FINAL = args.final

data = load_data("data/train.csv", PROCESSED_DATA)
trainX, trainY, testX, testY = get_train_test_split(data,
                                                    train_end="2013-07-31",
                                                    test_start="2013-08-01",
                                                    final=FINAL,
                                                    contamination=1e-4)

with open(REPORT) as json_file:
    report = json.load(json_file)

model = ModelLoader().get_model(model=report["model"], parameters=report["best_parameter"])
# parameters = report["best_parameter"]
# model.pass_arguments(parameters)



model.fit(trainX, trainY)
training_pmse = f"Training root mean square percentage error {metric(model.predict(trainX), trainY.values)}"
print(training_pmse)

if not FINAL:
    test_pmse = f"Test root mean square percentage error {metric(model.predict(testX), testY.values)}"
    print(test_pmse)


joblib.dump(model, "models/final/" + report["model"] + ".joblib")
