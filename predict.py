import argparse
from clean_database import DataCleaner
import joblib
from metric import metric
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Predict sales on rossman dataset')
parser.add_argument('--data', type=str, help='Path to data set, e.g. train.csv or holdout.csv')
parser.add_argument('--store', type=str, help='Path to store specific data, e.g. store.csv')

args = parser.parse_args()

# print(args.data, args.store)

cleaner = DataCleaner( args.data, args.store )
data = cleaner.get_clean_data()

print("Loading model...")

pipe = joblib.load("model.joblib")

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



