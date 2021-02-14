export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

train:
	python train.py --processed_data=processed_data.joblib

predict:
	python predict.py --data=data/holdout.csv --store=data/store.csv

setup:
	pip install -r requirements.txt
	python data.py --test 1
	# download pre-trained model
	wget https://www.dropbox.com/s/usioui56gvgt0ha/model.joblib

clean:
	rm *joblib
	rm data/*
