from __future__ import annotations

from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_PATH = DATA_DIR / "city_day.csv"
MODEL_PATH = BASE_DIR / "Saved_models" / "lgbm_aqi_model.joblib"
TARGET_COLUMN = "AQI"
FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]


def get_kaggle_data_path() -> Path:
    if not LOCAL_DATA_PATH.exists():
        dataset_path = Path(kagglehub.dataset_download("rohanrao/air-quality-data-in-india"))
        source_path = dataset_path / "city_day.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found in Kaggle download: {dataset_path}")
        LOCAL_DATA_PATH.write_bytes(source_path.read_bytes())
    return LOCAL_DATA_PATH


def main() -> None:
    data_path = get_kaggle_data_path()
    df = pd.read_csv(data_path)
    df = df.drop(columns=["City", "Date"], errors="ignore")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Features used:", FEATURE_COLUMNS)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop feature importance:")
    print(importance.sort_values(ascending=False).head(10).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
