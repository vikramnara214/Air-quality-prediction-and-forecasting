from __future__ import annotations

from pathlib import Path

import pandas as pd
from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parent
CITY_PATH = BASE_DIR / "Forecast" / "data" / "org_ch.csv"
OUTPUT_PATH = BASE_DIR / "Forecast" / "forecast_predictions.csv"


def main() -> None:
    city_df = pd.read_csv(CITY_PATH)
    city_df = city_df[["dates", "AQI"]].rename(columns={"dates": "ds", "AQI": "y"})
    city_df["ds"] = pd.to_datetime(city_df["ds"], errors="coerce")
    city_df = city_df.dropna().sort_values("ds").reset_index(drop=True)

    model = Prophet()
    model.fit(city_df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(output.to_string(index=False))
    print(f"\nForecast saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
