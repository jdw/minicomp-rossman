import argparse
import os
import shutil
import zipfile

import numpy as np
import pandas as pd


def mask_training_data(data):
    print('missing vals before {}'.format(np.sum(np.sum(pd.isnull(data.values)))))
    arr = data.values
    mask = np.random.choice([0, 1], p=[0.97, 0.03], size=arr.size).astype(bool).reshape(arr.shape)

    arr[mask] = np.nan
    masked = pd.DataFrame(arr, index=data.index, columns=data.columns)
    print('missing vals after {}'.format(np.sum(np.sum(masked.isnull()))))
    print('')
    return masked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=0, nargs='?')
    args = parser.parse_args()

    with zipfile.ZipFile('./data/rossmann-store-sales.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/raw')

    raw = pd.read_csv('./data/raw/train.csv', parse_dates=True)
    data = raw.set_index('Date')
    data = data.sort_index()

    train_end = '2014-07-31'
    test_start = '2014-08-01'
    train = data.loc[:train_end, :]
    test = data.loc[test_start:, :]
    assert train.shape[0] + test.shape[0] == raw.shape[0]

    np.random.seed(42)
    masked = mask_training_data(train)

    masked.to_csv('./data/train.csv')

    os.rename('./data/raw/store.csv', './data/store.csv')
    shutil.rmtree('./data/raw')

    if bool(int(args.test)):
        test.to_csv('./data/holdout.csv')
