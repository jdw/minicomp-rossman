## Rossman Kaggle Mini-Competition

This mini competition is adapted from the Kaggle Rossman challenge.  Please refrain from looking at the challenge on Kaggle until after you have finished - this will allow you to get a true measurement of where you are at as a data scientist.

## Usage
```bash
conda activate mini-comp-test

# Installs requirements, prepares Rossman data and downloads pretrained model
make setup

# Runs predictment against model
make predict
```

## Dataset

The dataset is made of two csvs:

```
#  store.csv
['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#  train.csv
['Date', 'Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday']
```

More info from Kaggle:

```
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```

The holdout test period is from 2014-08-01 to 2015-07-31 - the holdout test dataset is the same format as `train.csv`, as is called `holdout.csv`.

After running `python data.py -- test 1`, the folder `data` will look like:

```bash
data
├── holdout.csv
├── rossmann-store-sales.zip
├── store.csv
└── train.csv
```

## Scoring Criteria

The competition is scored based on a composite of predictive accuracy and reproducibility.

## Predictive accuracy

The task is to predict the `Sales` of a given store on a given day.

Submissions are evaluated on the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```

Zero sales days are ignored in scoring - part of your pipeline should look for these rows and drop them (in both test & train)

The team scores will be ranked - the highest score (lowest RMSPE) will receive a score of 10 for the scoring criteria section.

Each lower score (higher RMSPE) will receive a score of 10-(1 * number in ranking). If they are ranked second, score will be 10-2 = 8.

## Reproducibility

The entire model should be completely reproducible - to score this the teacher will clone your repository and follow the instructions as per the readme.  All teams start out with a score of 10.  One point is deducted for each step not included in the repo.

## Advice

Commit early and often

Notebooks don't merge easily!

Visualize early

Look at the predictions your model is getting wrong - can you engineer a feature for those samples?

Models
- baseline (average sales per store from in training data)
- random forest
- XGBoost

Use your DSR instructor(s)
- you are not alone - they are here to help with both bugs and data science advice
- git issues, structuring the data on disk, models to try, notebook problems and conda problems are all things we have seen before
