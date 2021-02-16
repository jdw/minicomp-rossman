import joblib
import pandas as pd
from src.utility.load_data import load_data


data = load_data("../data/holdout.csv", "../data/processed/clean_holdout_data.joblib")
predictions = data[["Sales", "Store"]]

for model in ["extratreepipe", "meanlazy", "randomforestpipe", "xgboostpipe", "prophet"]:
    pipe = joblib.load("../models/final/" + model + ".joblib")
    pred = pipe.predict(data.drop('Sales', axis=1))
    predictions.insert(len(predictions.columns), model, pred)

joblib.dump(predictions, "../data/processed/predictions.joblib")
