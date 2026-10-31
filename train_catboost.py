from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "data.csv"
MODEL_PATH = BASE_DIR / "Saved_models" / "catboost_pm25_model.joblib"
TARGET_COLUMN = "PM25"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.01,
        depth=10,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=55,
        od_type="Iter",
        od_wait=20,
        verbose=0,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Features used:", feature_columns)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    importance = pd.Series(model.get_feature_importance(), index=feature_columns)
    print("\nTop feature importance:")
    print(importance.sort_values(ascending=False).head(10).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
