import argparse
from clean_database import DataCleaner
import joblib
from metric import metric

parser = argparse.ArgumentParser(description='Predict sales on rossman dataset')
parser.add_argument('--data', type=str, help='Path to data set, e.g. train.csv or holdout.csv')
parser.add_argument('--store', type=str, help='Path to store specific data, e.g. store.csv')

args = parser.parse_args()

# print(args.data, args.store)

cleaner = DataCleaner( args.data, args.store )
data = cleaner.get_clean_data()

pipe = joblib.load("model.joblib")

print(f"Test percentage mean squared error {metric( pipe.predict(data.drop('Sales', axis=1)), data['Sales'].values )}")
