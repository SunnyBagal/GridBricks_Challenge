import os
import pandas as pd

UK_TIMEZONE = "Europe/London"
DEFAULT_DECISION_TIME = "10:00"
FIRST_DELIVERY_DAY = "2026-01-02"
LAST_DELIVERY_DAY = "2026-02-01"
HORIZON_BUCKET_HOURS = 6
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")


def load_actuals(path=os.path.join(DATA_FOLDER, "actuals.csv")):
    df = pd.read_csv(path)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df


def load_forecasts(path=os.path.join(DATA_FOLDER, "forecasts.csv")):
    df = pd.read_csv(path)
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df


def get_decision_time_utc(delivery_date, decision_time=DEFAULT_DECISION_TIME):
    day_before = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
    uk_time = pd.Timestamp(f"{day_before.date()} {decision_time}").tz_localize(UK_TIMEZONE)
    return uk_time.tz_convert("UTC")


def get_delivery_day_window_utc(delivery_date):
    day = pd.Timestamp(delivery_date)
    start = day.tz_localize(UK_TIMEZONE).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(UK_TIMEZONE).tz_convert("UTC")
    return start, end


def get_actuals_for_day(actuals, delivery_date):
    day_start, day_end = get_delivery_day_window_utc(delivery_date)
    in_day = (actuals["startTime"] >= day_start) & (actuals["startTime"] < day_end)
    return actuals[in_day].sort_values("startTime").reset_index(drop=True)


def select_decision_forecast(forecasts, delivery_date, decision_time=DEFAULT_DECISION_TIME):
    decision_time_utc = get_decision_time_utc(delivery_date, decision_time)
    day_start, day_end = get_delivery_day_window_utc(delivery_date)

    available = forecasts[forecasts["publishTime"] <= decision_time_utc]

    in_day = (available["startTime"] >= day_start) & (available["startTime"] < day_end)
    available = available[in_day]

    available = available.sort_values("publishTime")
    latest = available.drop_duplicates(subset="startTime", keep="last")
    return latest.sort_values("startTime").reset_index(drop=True)


def build_decision_errors(
    actuals,
    forecasts,
    decision_time=DEFAULT_DECISION_TIME,
    first_day=FIRST_DELIVERY_DAY,
    last_day=LAST_DELIVERY_DAY,
):
    daily_forecasts = []
    for delivery_date in pd.date_range(first_day, last_day, freq="D"):
        daily_forecasts.append(select_decision_forecast(forecasts, delivery_date, decision_time))
    decision_forecasts = pd.concat(daily_forecasts)

    decision_forecasts = decision_forecasts.rename(columns={"generation": "forecast"})
    actual_values = actuals.rename(columns={"generation": "actual"})

    errors = decision_forecasts.merge(actual_values, on="startTime", how="inner")
    errors["error"] = errors["forecast"] - errors["actual"]
    return errors


def compute_metrics(errors):
    error = errors["error"]
    if len(error) == 0:
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "p10": None, "p50": None, "p90": None}

    return {
        "count": int(len(error)),
        "mae": float(error.abs().mean()),
        "rmse": float((error ** 2).mean() ** 0.5),
        "bias": float(error.mean()),
        "p10": float(error.quantile(0.10)),
        "p50": float(error.quantile(0.50)),
        "p90": float(error.quantile(0.90)),
    }


def error_by_hour_of_day(errors):
    errors = errors.copy()
    errors["hour"] = errors["startTime"].dt.tz_convert(UK_TIMEZONE).dt.hour

    rows = []
    for hour, group in errors.groupby("hour"):
        metrics = compute_metrics(group)
        metrics["hour"] = int(hour)
        rows.append(metrics)
    return rows


def error_by_horizon(actuals, forecasts, first_day=FIRST_DELIVERY_DAY, last_day=LAST_DELIVERY_DAY):
    period_start, _ = get_delivery_day_window_utc(first_day)
    _, period_end = get_delivery_day_window_utc(last_day)
    in_period = (forecasts["startTime"] >= period_start) & (forecasts["startTime"] < period_end)

    forecast_values = forecasts[in_period].rename(columns={"generation": "forecast"})
    actual_values = actuals.rename(columns={"generation": "actual"})
    merged = forecast_values.merge(actual_values, on="startTime", how="inner")
    merged["error"] = merged["forecast"] - merged["actual"]

    merged["horizon_hours"] = (merged["startTime"] - merged["publishTime"]).dt.total_seconds() / 3600
    
    merged = merged[merged["horizon_hours"] > 0]

    merged["bucket_start"] = (merged["horizon_hours"] // HORIZON_BUCKET_HOURS * HORIZON_BUCKET_HOURS).astype(int)

    rows = []
    for bucket_start, group in merged.groupby("bucket_start"):
        metrics = compute_metrics(group)
        metrics["horizon_from"] = int(bucket_start)
        metrics["horizon_to"] = int(bucket_start) + HORIZON_BUCKET_HOURS
        rows.append(metrics)
    return rows
