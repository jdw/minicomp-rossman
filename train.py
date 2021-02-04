from metric import metric
from metric import sklearn_metric

import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

import sklearn.ensemble as se
import sklearn

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

from clean_database import DataCleaner

from mean_lazy_estimator import MeanLazyEstimator
from last_lazy_estimator import LastLazyEstimator

import xgboost as xg

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import joblib

import subprocess

import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

PROCESSED_DATA = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
subprocess = subprocess.Popen("echo \"$(git rev-parse HEAD)-$(date '+%Y.%m.%d-%H.%M.%S')\"", shell=True, stdout=subprocess.PIPE)
gitRevId_timestamp = subprocess.stdout.read().decode("utf-8").strip()
output_file = open("results/" + gitRevId_timestamp + "-" +OUTPUT_FILE, "a")
FINAL = True

data = ""
try:
    data = joblib.load(PROCESSED_DATA)
except FileNotFoundError:
    dc = DataCleaner("data/train.csv")
    data = dc.get_clean_data()
    joblib.dump(data, PROCESSED_DATA)

if FINAL:
    trainX = data.drop("Sales", axis=1)
    trainY = data.Sales
else:
    trainX = data.loc[:"2013-07-31"]
    testX = data.loc["2013-08-01":]

    trainY = trainX["Sales"]
    trainX.drop("Sales", axis=1, inplace=True)

    testY = testX["Sales"]
    testX.drop("Sales", axis=1, inplace=True)

ct = ColumnTransformer([
        ( "target_enc", ce.target_encoder.TargetEncoder(cols=["Store", 'StoreType', 'Assortment', 'PromoInterval']), ["Store", 'StoreType', 'Assortment', 'PromoInterval'] ),
        ("one_hot_enc", OneHotEncoder(handle_unknown="ignore"), ['StateHoliday'])
    ], remainder='passthrough')

estimators = [
    # ("rfr", RandomForestRegressor(n_jobs=-1) ),
    ("rfr25", RandomForestRegressor(max_depth=25, n_jobs=-1) ),
    ("rfr30", RandomForestRegressor(max_depth=30, n_jobs=-1) ),
    ("rfr35", RandomForestRegressor(max_depth=35, n_jobs=-1) ),
    ("rfr50", RandomForestRegressor(max_depth=50, n_jobs=-1) ),
    ("xtr30", se.ExtraTreesRegressor(n_jobs=-1, max_depth=30)),
    ("xtr35", se.ExtraTreesRegressor(n_jobs=-1, max_depth=35)),
    ("xtr25", se.ExtraTreesRegressor(n_jobs=-1, max_depth=25))
]

pipe = Pipeline([
    ("pre", ct),
    # ("rf", RandomForestRegressor(n_jobs=-1) )
    ("reg", se.ExtraTreesRegressor(n_jobs=-1, max_depth=19,n_estimators=200)),
    #("reg", se.VotingRegressor(estimators=estimators, verbose=True) )
])


pipe.fit(trainX, trainY)

if not FINAL:
    training_pmse = f"Training percentage mean squared error {metric( pipe.predict(trainX), trainY.values )}"
    test_pmse = f"Test percentage mean squared error {metric( pipe.predict(testX), testY.values )}"

    print(training_pmse)
    print(test_pmse)

    output_file.write(training_pmse)
    output_file.write('\n')
    output_file.write(test_pmse)
    output_file.close()

joblib.dump(pipe, "model.joblib")
