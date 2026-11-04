PYTHON ?= python

.PHONY: all data notebook test app

all: data notebook test

data:
	$(PYTHON) -m src.run_all

notebook:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb notebooks/02_drivers.ipynb notebooks/03_forecast.ipynb notebooks/04_scenarios.ipynb

test:
	$(PYTHON) -m pytest -q

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
