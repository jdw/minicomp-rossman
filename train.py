from metric import metric

import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

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

data = ""
try:
    data = joblib.load(PROCESSED_DATA)
except FileNotFoundError:
    dc = DataCleaner("data")
    data = dc.get_clean_data()
    joblib.dump(data, PROCESSED_DATA)



# print(data.isnull().any())

trainX = data.loc[:"2013-07-31"]
testX = data.loc["2013-08-01":]

trainY = trainX["Sales"]
trainX.drop("Sales", axis=1, inplace=True)

testY = testX["Sales"]
testX.drop("Sales", axis=1, inplace=True)

ct = ColumnTransformer([
        ( "target_enc", ce.target_encoder.TargetEncoder(cols=["Store"]), ["Store"] ),
       #  ('scaler', StandardScaler(), ['CompetitionDistance',
       # 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       # 'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month',
       # 'Day']),
        ("one_hot_enc", OneHotEncoder(handle_unknown="ignore"), ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])
    ], remainder='passthrough')

pipe = Pipeline([
    ("pre", ct),
    # ("imputer", KNNImputer()),
    ("rf", RandomForestRegressor(verbose=1, n_jobs=16) )
])

# pipe = Pipeline([
#     ("pre", ct),
#     ("rf", xg.XGBRegressor(n_estimators = 1000) )
# ])

pipe.fit(trainX, trainY)

training_pmse = f"Training percentage mean squared error {metric( pipe.predict(trainX), trainY.values )}"
test_pmse = f"Test percentage mean squared error {metric( pipe.predict(testX), testY.values )}"

print(training_pmse)
print(test_pmse)

output_file.write(training_pmse)
output_file.write('\n')
output_file.write(test_pmse)
output_file.close()


# pipe_lazy_mean = Pipeline([
#     ("pre", ct),
#     ("lm", MeanLazyEstimator() )
# ])
#
#
# pipe_lazy_last = Pipeline([
#     ("pre", ct),
#     ("ll", LastLazyEstimator() )
# ])
#
# print("Lazy mean:")
# pipe_lazy_mean.fit(trainX, trainY)
# print(f"Training percentage mean squared error {metric( pipe_lazy_mean.predict(trainX), trainY.values )}")
# print(f"Test percentage mean squared error {metric( pipe_lazy_mean.predict(testX), testY.values )}")
#
# print("Lazy last:")
# pipe_lazy_last.fit(trainX, trainY)
# print(f"Training percentage mean squared error {metric( pipe_lazy_last.predict(trainX), trainY.values )}")
# print(f"Test percentage mean squared error {metric( pipe_lazy_last.predict(testX), testY.values )}")
