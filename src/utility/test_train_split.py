import sklearn.ensemble as se

def get_train_test_split( data, train_end="2013-07-31", test_start="2013-08-01", final=False, contamination=0.01 ):
    if final:
        data = remove_outlier(data, contamination=contamination)
        trainX = data.drop("Sales", axis=1)
        trainY = data.Sales
    else:
        trainX = data.loc[:"2013-07-31"]
        testX = data.loc["2013-08-01":]

        trainX = remove_outlier(trainX, contamination=contamination)

        trainY = trainX["Sales"]
        trainX.drop("Sales", axis=1, inplace=True)

        testY = testX["Sales"]
        testX.drop("Sales", axis=1, inplace=True)

    return trainX, trainY, testX, testY

def remove_outlier(data, contamination=0.01):
    outlier_map = se.IsolationForest(contamination=contamination).fit_predict(
        data[["Sales", "CompetitionDistance", "aggregated_promo2", "days_since_competition"]])
    return data[outlier_map == 1]