import os
import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

ACTUALS_FROM_DATE = "2026-01-01"
ACTUALS_TO_DATE = "2026-02-02"
FORECASTS_PUBLISHED_FROM = "2025-12-28T00:00:00Z"
FORECASTS_PUBLISHED_TO = "2026-02-02T00:00:00Z"


def download_json(url, params):
    print("Downloading", url, params)
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    print("  got", len(rows), "rows")
    return rows

def download_actuals():
    rows = download_json(
        BASE_URL + "/FUELHH/stream",
        {
            "settlementDateFrom": ACTUALS_FROM_DATE,
            "settlementDateTo": ACTUALS_TO_DATE,
            "fuelType": "WIND",
        },
    )
    df = pd.DataFrame(rows)
    df = df[["startTime", "generation"]]
    df = df.drop_duplicates(subset="startTime", keep="last")
    df = df.sort_values("startTime")
    return df


def download_forecasts():
    rows = download_json(
        BASE_URL + "/WINDFOR/stream",
        {
            "publishDateTimeFrom": FORECASTS_PUBLISHED_FROM,
            "publishDateTimeTo": FORECASTS_PUBLISHED_TO,
        },
    )

    df = pd.DataFrame(rows)
    df = df[["publishTime", "startTime", "generation"]]
    df = df.drop_duplicates(subset=["publishTime", "startTime"], keep="last")
    df = df.sort_values(["publishTime", "startTime"])
    return df


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    actuals = download_actuals()
    actuals_path = os.path.join(DATA_FOLDER, "actuals.csv")
    actuals.to_csv(actuals_path, index=False)
    print("Saved", len(actuals), "rows to", actuals_path)

    forecasts = download_forecasts()
    forecasts_path = os.path.join(DATA_FOLDER, "forecasts.csv")
    forecasts.to_csv(forecasts_path, index=False)
    print("Saved", len(forecasts), "rows to", forecasts_path)


if __name__ == "__main__":
    main()
