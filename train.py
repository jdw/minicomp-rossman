from metric import metric
from metric import sklearn_metric

import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

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

dc = DataCleaner("data")
data = dc.get_clean_data()
# joblib.dump(data, "processed_data.joblib")
# data = joblib.load("processed_data.joblib")

# print(data.isnull().any())

trainX = data.loc[:"2013-07-31"]
testX = data.loc["2013-08-01":]

trainY = trainX["Sales"]
trainX.drop("Sales", axis=1, inplace=True)

testY = testX["Sales"]
testX.drop("Sales", axis=1, inplace=True)

ct = ColumnTransformer([
        ( "target_enc", ce.target_encoder.TargetEncoder(cols=["Store", 'StoreType', 'Assortment', 'PromoInterval']), ["Store", 'StoreType', 'Assortment', 'PromoInterval'] ),
       #  ('scaler', StandardScaler(), ['CompetitionDistance', 'Year', 'Month',
       # 'Day']),
        ("one_hot_enc", OneHotEncoder(handle_unknown="ignore"), ['StateHoliday'])
       #  ("one_hot_enc", OneHotEncoder(handle_unknown="ignore"), ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])
    ], remainder='passthrough')

pipe = Pipeline([
    ("pre", ct),
    # ("imputer", KNNImputer()),
    ("rf", RandomForestRegressor(n_jobs=-1) )
])

param_grid = {
    "rf__n_estimators": [10,50,100],
    "rf__max_depth": [10,20, 30, 40, 50],
    "rf__min_impurity_decrease": [0.]
}

parameters = ParameterGrid(param_grid)
for p in parameters:
    pipe = Pipeline([
        ("pre", ct),
        # ("imputer", KNNImputer()),
        ("rf", RandomForestRegressor(n_jobs=16,
                                     n_estimators=p["rf__n_estimators"],
                                     max_depth=p["rf__max_depth"],
                                     min_impurity_decrease=p["rf__min_impurity_decrease"]
                                     ))
    ])
    pipe.fit(trainX, trainY)
    report = ""
    for key, value in p.items():
           report += f'{key}={value}, '
    report += f'train score {metric( pipe.predict(trainX), trainY.values )}, test score {metric( pipe.predict(testX), testY.values )}'
    print(report)
    # print(f'n_estimators={p["rf__n_estimators"]}, max_depth={p["rf__max_depth"]}, ccp_alpha={p["rf__ccp_alpha"]}: train score {metric( pipe.predict(trainX), trainY.values )}, test score {metric( pipe.predict(testX), testY.values )}')

# param_grid = {
#     "max_depth": [1,2,4,6,8,10],
#     "min_child_weight": [1,5,10],
# }
#
# parameters = ParameterGrid(param_grid)
# for p in parameters:
#     pipe = Pipeline([
#         ("pre", ct),
#         ("rf", xg.XGBRegressor(n_estimators= 1000,
#                                max_depth = p["max_depth"],
#                                min_child_weight=p["min_child_weight"],
#                                # gamma=50,
#                                # subsample=0.8,
#                                # colsample_bytree=0.8,
#                                booster="gbtree",
#                                # objective="reg:squarederror",
#                                # reg_alpha = 1000.,
#                                # reg_lambda = 1000,
#                                # tree_method="hist",
#                                # subsample=0.8,
#                                learning_rate=0.5,
#                                n_jobs=-1) )
#     ])
#
#     pipe.fit(trainX, trainY)
#     print(f'max_depth = {p["max_depth"]}, min_child_weight = {p["min_child_weight"]}; train: {metric( pipe.predict(trainX), trainY.values )}, test: {metric( pipe.predict(testX), testY.values )}')
# # print(f"Training percentage mean squared error {metric( pipe.predict(trainX), trainY.values )}")
# # print(f"Test percentage mean squared error {metric( pipe.predict(testX), testY.values )}")


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