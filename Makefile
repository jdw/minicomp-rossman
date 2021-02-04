export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

run:
	python train.py processed_data.joblib results.txt

predict:
	python predict.py --data=data/holdout.csv --store=data/store.csv

setup:
	pip install -r requirements.txt
	python data.py --test 1
	
clean:
	rm *joblib
	rm data/*
