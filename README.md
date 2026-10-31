# Air Pollution Forecasting and Prediction

This project is implemented in Python and uses machine learning models to predict AQI and forecast pollution trends.

## Models

- XGBoost for AQI regression
- CatBoost for PM2.5 and related feature modeling
- Prophet for time-series forecasting
- Streamlit dashboard for interactive prediction

## Features

- Temperature
- Wind speed
- Pressure
- NO2 concentration
- Rainfall
- PM10
- PM25
- AQI

## Python stack

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- prophet
- joblib

## Project scripts

- `clean_data.py` – preprocesses the raw dataset
- `train_xgboost.py` – trains the XGBoost AQI model
- `train_catboost.py` – trains the CatBoost model
- `forecast_prophet.py` – generates a 7-day forecast with Prophet
- `app.py` – Streamlit dashboard for AQI prediction

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

Legacy R project files and R-specific model artifacts were removed in favor of the Python implementation.
