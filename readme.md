# Rossman Sales Prediction (Mini-Competition)

This project is the result of a small competition ran at the "Data Science Retreat" bootcamp. The competition challenged small teams to predict the sales from the Rossman dataset. The winner was determined not only by the final performance but also the reproducebility and cleaness of the code.

Since the conclusion of the challenge, we have gone back to implement some of the ideas from the other teams, clean up the code and make this project more presentable.

## Results
According to the requirements of the competition, we score the models using the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

Thus, lower score is better.

|Model   | RMSPE   |
|---|---|
|Mean Lazy Estimator| 41.95|
|Prophet|31.39|
|ExtraTreeRegressor| 28.33|
|RandomForestRegressor| 28.86|
|XGBoostRegressor| 26.69 |

## Usage
### Optional: Setup conda environment
Create a new conda environment:
```bash
conda create -n mini-comp-test python=3.8
```
and activate the environment:
```bash
conda activate mini-comp-test
```
### Setup
Clone the repository and change directory to it. Then execute:
```bash
make setup
```
This will install the requirements, prepare the Rossman data as well as download the pretrained model.

### Hyperparameter search
We have implemented a random grid search to find the best hyperparameters.
This procedure is controled by a instructions file, of the following format:

```json
{
  "train": "data/train.csv",
  "model": "xgboostpipe",
  "model_output": "models/xgboostpipe_fine.joblib",
  "report_output": "results/xgboostpipe_fine.json",
  "grid": {
    "gamma": [0],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    "max_depth": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "n_estimators": [10,20,30,40,50],
    "reg_alpha": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150 ],
    "reg_lambda": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
  }
}
```

To begin the hyperparameter search simply execute

```bash
make hyperparameter-search INSTRUCTION=<path to instruction> MAXRUNS=<number of max runs>
```

This will train a number of random parameter combinations from the provided grid, keep track of the best model and finally save it to the location specified in the instructions file.

### Final training
In order to perform the final training on the best model and the whole data set, execute

```bash
make train REPORT=<path to report> FINAL=1
```

This will perform the training on the whole data set and save the model to ```m̀odels/final/<model name>.joblib```.

### Run the predictions
To run the predictions on the holdout data, simply run:
```bash
make predict MODEL=<path to model>
```
After printing the final score, the script will offer you to plot the predictions against the real values, to monitor the performance.

# Modeling Approach
## Original Dataset
The data set and goal is described in detail at the [original repository](https://github.com/ADGEfficiency/minicomp-rossman).
## Data Exploration and Cleaning
We first monitor the correlation and distribution of the available features, as well as the target "Sales". Given this information we clean and prepare the data.

### Correlations
We find that the "Open" and "Customers" features are strongly correlated with the target.
- The "Open" feature indicates days where the store closed and reported zero sales, due to the nature of the required metric, we drop this feature and entries with zero sales respectively.
- The "Customers" feature directly indicates the amount of sales, and is not expected to be known in advance. At the current stage we drop this column. In the future this column might be used to engineer better features.
- 
### Null Values
We find multiple columns that contain null values, which we fill with the mean or mode as follows:
- mean: ``"CompetitionDistance"``
- mode: ``"SchoolHoliday"``, ``"StateHoliday"``, ``"Promo"``, ``"DayOfWeek"``

### Outliers
We employ an ``IsolationForest`` to detect and remove outliers from the training set before training. The contamination is set to ``1e-4``

### Feature Engineering
We have designed and implemented three new features, extracted from the available data:
- The columns ``"Promo2"``, ``"Prome2SinceYear"`` and ``"Promo2SinceWeek"`` are aggregated into a single numerical column counting the days since the continuous promotion started. If there is no current running promotion the value is set to ``0``. The original columns are dropped.
- Similarly, the columns ``"CompeitionOpenSinceMonth`` and ``"CompetitionOpenSinceYear"`` are aggregated into a single feature counting the days since competition opened.
- Noting that days around the weekend score the highest average sales, we generate a new feature flagging fridays, saturdays and mondays.
- In addition: We have flagged high sales days of the year by manually monitoring the mean sales per day of year. These days correspond to days around eastern, christmas and the likes.

### Categorical Encoding
All non numerical columns are encoded to resemble numerical values, in the following manner:
- The ``"Store"``, ``"StoreType"``,``"Assortment"``,``"PromoInterval"`` are target encoded, reflecting the mean ``"Sales"`` per category.
- The ``"StateHoliday"`` column is one-hot encoded.

# Models and Training
For all training runs, we split the data such that the validation set contains the last year of data, matching the time period contained in the holdout set.

## Models
We then proceed to implement, train and test a multitude of different regression models, including custom lazy estimators, ``XGBoostRegressor``, ``RandomForestRegressor``, and ``ExtraTreeRegressor``. In addition we explore an alternative approach provided by facebook's ``Prophet``.
Following the procedure described above.

# Future Improvements
The current root mean squared percentage error is not particularly impressive. In order to improve it, one should focus a lot more on the features, performing manual outlier removal, incorporate the customer data and more.