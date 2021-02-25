export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

MODEL="model/final/xgboostpipe.joblib"
PROCESSED_DATA="data/processed/clean_data.joblib"
REPORT="results/xgboostpipe_fine.json"
INSTRUCTION="instructions/xgboostpipe_fine.json"
MAXRUNS=100
FINAL=1

train:
	python -m src.train --processed-data=$(PROCESSED_DATA) --report=$(REPORT) --final=$(FINAL)

hyperparameter-search:
	python -m src.train_random_param --instructions=$(INSTRUCTION) --max_runs=$(MAXRUNS)

predict:
	python -m src.predict --data=data/holdout.csv --model=$(MODEL)

setup:
	pip install -r requirements.txt
	python data.py --test 1

clean:
	rm *joblib
	rm data/*
