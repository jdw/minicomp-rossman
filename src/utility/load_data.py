from src.data_cleaner import DataCleaner
import joblib


def load_data(path_to_csv, path_to_joblib="data/processed/clean_data.joblib"):
    try:
        data = joblib.load(path_to_joblib)
    except FileNotFoundError:
        dc = DataCleaner(path_to_csv)
        data = dc.get_clean_data()
        joblib.dump(data, path_to_joblib)
    return data
