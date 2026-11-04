# Revenue Forecast & Scenario Planner

> Demonstration project using public data (Olist). Not affiliated with any employer.

**Business question:** What revenue should we plan for the next 2 quarters, and which markets or
categories should get extra growth investment?

**Answer:** see the [decision memo](docs/decision_memo.md). Work in progress; the full case study comes in Stage 6.

## Run it

Download the Olist CSVs from Kaggle into `data/raw/`, then:

```bash
pip install -r requirements.txt
python -m src.run_all                 # tables, drivers, forecast, scenarios -> data/processed/
python -m pytest -q                   # data, driver, forecast, scenario and app tests
streamlit run app/streamlit_app.py    # interactive scenario planner
```

Notebooks `notebooks/01`–`04` walk through exploration, drivers, forecast and scenarios.

Data: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
licensed CC BY-NC-SA 4.0. Credit: Olist.
