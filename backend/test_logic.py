import math
import pandas as pd

import logic


def make_forecasts(rows):
    df = pd.DataFrame(rows, columns=["publishTime", "startTime", "generation"])
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df


def make_actuals(rows):
    df = pd.DataFrame(rows, columns=["startTime", "generation"])
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df


def test_decision_time_in_winter_is_same_as_utc():
    result = logic.get_decision_time_utc("2026-01-10")
    assert result == pd.Timestamp("2026-01-09T10:00:00Z")


def test_decision_time_in_summer_is_one_hour_earlier_in_utc():
    result = logic.get_decision_time_utc("2026-07-10")
    assert result == pd.Timestamp("2026-07-09T09:00:00Z")


def test_custom_decision_time():
    result = logic.get_decision_time_utc("2026-01-10", "15:30")
    assert result == pd.Timestamp("2026-01-09T15:30:00Z")


def test_ignores_forecasts_published_after_decision_time():
    forecasts = make_forecasts([
        ["2026-01-09T09:30:00Z", "2026-01-10T12:00:00Z", 100],
        ["2026-01-09T11:30:00Z", "2026-01-10T12:00:00Z", 999],
    ])
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert list(result["generation"]) == [100]


def test_picks_latest_forecast_before_decision_time():
    forecasts = make_forecasts([
        ["2026-01-09T04:30:00Z", "2026-01-10T12:00:00Z", 50],
        ["2026-01-09T09:30:00Z", "2026-01-10T12:00:00Z", 100],
        ["2026-01-09T07:30:00Z", "2026-01-10T12:00:00Z", 70],
    ])
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert list(result["generation"]) == [100]


def test_forecast_published_exactly_at_decision_time_is_used():
    forecasts = make_forecasts([
        ["2026-01-09T09:30:00Z", "2026-01-10T12:00:00Z", 100],
        ["2026-01-09T10:00:00Z", "2026-01-10T12:00:00Z", 200],
    ])
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert list(result["generation"]) == [200]


def test_only_returns_hours_inside_delivery_day():
    forecasts = make_forecasts([
        ["2026-01-09T09:30:00Z", "2026-01-09T23:00:00Z", 1],
        ["2026-01-09T09:30:00Z", "2026-01-10T00:00:00Z", 2],
        ["2026-01-09T09:30:00Z", "2026-01-10T23:00:00Z", 3],
        ["2026-01-09T09:30:00Z", "2026-01-11T00:00:00Z", 4],
    ])
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert list(result["generation"]) == [2, 3]


def test_uses_older_forecast_when_latest_run_misses_an_hour():
    forecasts = make_forecasts([
        ["2026-01-09T04:30:00Z", "2026-01-10T12:00:00Z", 50],
        ["2026-01-09T04:30:00Z", "2026-01-10T13:00:00Z", 60],
        ["2026-01-09T09:30:00Z", "2026-01-10T12:00:00Z", 100],
    ])
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert list(result["generation"]) == [100, 60]


def test_real_data_uses_0930_forecast_for_10am_decision():
    forecasts = logic.load_forecasts()
    result = logic.select_decision_forecast(forecasts, "2026-01-10")
    assert len(result) == 24
    assert set(result["publishTime"]) == {pd.Timestamp("2026-01-09T09:30:00Z")}


def test_metrics_on_known_errors():
    errors = pd.DataFrame({"error": [10, -10, 20, -20]})
    metrics = logic.compute_metrics(errors)
    assert metrics["count"] == 4
    assert metrics["mae"] == 15
    assert math.isclose(metrics["rmse"], math.sqrt(250))
    assert metrics["bias"] == 0


def test_metrics_with_no_data_returns_none():
    metrics = logic.compute_metrics(pd.DataFrame({"error": []}))
    assert metrics["count"] == 0
    assert metrics["mae"] is None


def test_error_is_positive_when_forecast_too_high():
    forecasts = make_forecasts([["2026-01-09T09:30:00Z", "2026-01-10T12:00:00Z", 150]])
    actuals = make_actuals([
        ["2026-01-10T12:00:00Z", 100],
        ["2026-01-10T12:30:00Z", 999],  
    ])
    errors = logic.build_decision_errors(actuals, forecasts, first_day="2026-01-10", last_day="2026-01-10")
    assert list(errors["error"]) == [50]


def test_error_by_horizon_groups_hours_ahead_and_skips_past_hours():
    forecasts = make_forecasts([
        ["2026-01-10T09:30:00Z", "2026-01-10T08:00:00Z", 100],  
        ["2026-01-10T09:30:00Z", "2026-01-10T12:00:00Z", 110],  
        ["2026-01-10T09:30:00Z", "2026-01-10T22:00:00Z", 130],  
    ])
    actuals = make_actuals([
        ["2026-01-10T08:00:00Z", 100],
        ["2026-01-10T12:00:00Z", 100],
        ["2026-01-10T22:00:00Z", 100],
    ])
    rows = logic.error_by_horizon(actuals, forecasts, first_day="2026-01-10", last_day="2026-01-10")
    assert [(row["horizon_from"], row["bias"]) for row in rows] == [(0, 10), (12, 30)]
