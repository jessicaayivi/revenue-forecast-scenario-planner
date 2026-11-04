PYTHON ?= python

.PHONY: all data notebook test

all: data notebook test

data:
	$(PYTHON) -m src.run_all

notebook:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb notebooks/02_drivers.ipynb

test:
	$(PYTHON) -m pytest -q
