import argparse
from src.data_cleaner import DataCleaner
from src.utility.load_data import load_data
import joblib
from metric import metric
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Predict sales on rossman dataset')
parser.add_argument('--data', type=str, help='Path to data set, e.g. train.csv or holdout.csv', default="../data/holdout.csv")
parser.add_argument('--model', type=str, help='Path to model dump (joblib)', default="../models/final/meanlazy.joblib")

args = parser.parse_args()
#
data = load_data(args.data, "../data/processed/clean_holdout_data.joblib")

print("Loading model...")

pipe = joblib.load(args.model)

print("Predicting sales...")
pred = pipe.predict(data.drop('Sales', axis=1))
print(f"Test percentage mean squared error {metric( pred, data['Sales'].values )}")

unique_stores = data.Store.unique()
compare_df = data[["Sales"]].copy()
compare_df["Predict"] = pred
for u in unique_stores:
    response = input("Do you want to see a prediction? [y/N]:")
    if response == "y":
        map_store = (data.Store == u)
        compare_df[map_store].plot()
        plt.title(f"Store ID {u}")
        plt.show()
    else:
        break;



