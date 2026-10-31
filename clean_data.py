from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "data.csv"
OUTPUT_PATH = BASE_DIR / "Data" / "cleaned_air_quality.csv"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    cleaned = clean_dataframe(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
