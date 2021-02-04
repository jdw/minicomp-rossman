export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

run:
	#jupyter notebook --port=8080 --no-browser train.ipynb
	jupyter nbconvert --to python train.ipynb
	python train.py
