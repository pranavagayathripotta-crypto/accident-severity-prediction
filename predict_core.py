"""
Core prediction logic for the v3 accident risk model -- extracted from
the original accident_severity_pipeline_v3 script so the website can
reuse the exact same address/weather/risk-lookup logic.
"""
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_severity_model.joblib"
COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
CATEGORY_MAP_PATH = MODEL_DIR / "category_maps.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.joblib"
RISK_MAP_PATH = MODEL_DIR / "local_risk_map.joblib"
WEATHER_MEDIAN_PATH = MODEL_DIR / "weather_medians.joblib"

RISK_CELL_SIZE = 0.003
RISK_LABELS = {0: "Minor", 1: "Elevated risk (injury/fatality)"}
WEATHER_COLUMNS = ["Temperature_C", "Windspeed_kmh"]

# Must match the training script's config (BOROUGH / LAT_MIN.. etc in
# train_model.py) -- the model was only ever trained on accidents inside
# this box, so predictions outside it aren't scientifically meaningful.
MIDTOWN_LAT_MIN, MIDTOWN_LAT_MAX = 40.742, 40.767
MIDTOWN_LNG_MIN, MIDTOWN_LNG_MAX = -74.008, -73.971

MIN_VEHICLE_COUNT = 1
MAX_VEHICLE_COUNT = 10  # generous upper bound; the training data essentially never saw more than this

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_HOURLY_VARS = "temperature_2m,precipitation,snowfall,windspeed_10m,weathercode"

# Current/forecast weather is cached for this long before we re-fetch it.
# Historical weather (a past date/hour) never changes once recorded, so
# that lookup is cached indefinitely instead (see fetch_historical_weather).
CURRENT_WEATHER_TTL_SECONDS = 600

# Reusing one requests.Session (instead of a fresh connection per call)
# lets repeated calls to the same host reuse their TCP/TLS handshake,
# which noticeably cuts latency since every prediction can involve a
# geocode call plus a weather call.
_session = requests.Session()

_model = None
_feature_columns = None
_category_maps = None
_threshold = None
_risk_map = None
_weather_medians = None
_load_error = None


def get_resources():
    global _model, _feature_columns, _category_maps, _threshold, _risk_map, _weather_medians, _load_error
    if _model is None and _load_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _feature_columns = joblib.load(COLUMNS_PATH)
            _category_maps = joblib.load(CATEGORY_MAP_PATH) if CATEGORY_MAP_PATH.exists() else {}
            _threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.5
            _risk_map = joblib.load(RISK_MAP_PATH) if RISK_MAP_PATH.exists() else {"__global__": 0.0}
            _weather_medians = joblib.load(WEATHER_MEDIAN_PATH) if WEATHER_MEDIAN_PATH.exists() else {}
        except FileNotFoundError:
            _load_error = (
                "Model files not found. Copy best_severity_model.joblib, "
                "feature_columns.joblib, category_maps.joblib, decision_threshold.joblib, "
                "local_risk_map.joblib, and weather_medians.joblib into the models/ folder."
            )
    return _model, _feature_columns, _category_maps, _threshold, _risk_map, _weather_medians, _load_error


# ============================================================
# VALIDATION
# ============================================================

def validate_location(latitude: float, longitude: float) -> None:
    """Raises ValueError if the location isn't a real coordinate, or
    isn't inside the Midtown Manhattan box the model was trained on."""
    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    if not (MIDTOWN_LAT_MIN <= latitude <= MIDTOWN_LAT_MAX
            and MIDTOWN_LNG_MIN <= longitude <= MIDTOWN_LNG_MAX):
        raise ValueError(
            "This model is trained only on Midtown Manhattan accident data "
            f"(lat {MIDTOWN_LAT_MIN}-{MIDTOWN_LAT_MAX}, lon {MIDTOWN_LNG_MIN}-{MIDTOWN_LNG_MAX}). "
            "Please enter a location within the supported area."
        )


def validate_vehicle_count(vehicle_count: int) -> None:
    if vehicle_count < MIN_VEHICLE_COUNT:
        raise ValueError(f"Vehicle count must be at least {MIN_VEHICLE_COUNT}.")
    if vehicle_count > MAX_VEHICLE_COUNT:
        raise ValueError(f"Vehicle count must be {MAX_VEHICLE_COUNT} or fewer.")


# ============================================================
# GEOCODING & WEATHER
# ============================================================
#
# Both geocoding and weather hit external APIs over the network, which
# is by far the slowest part of handling a prediction request. Repeat
# visitors very often ask about the same handful of addresses/blocks,
# so caching here cuts both latency and the number of outbound calls.

def _round_coord(value: float) -> float:
    """Round to ~111m grid cells (matches RISK_CELL_SIZE's precision) so
    nearby requests share a weather cache entry instead of each missing."""
    return round(value, 3)


@lru_cache(maxsize=256)
def _geocode_address_cached(address_key: str):
    try:
        resp = _session.get(
            GEOCODE_URL,
            params={"name": address_key, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        label_parts = [r.get("name"), r.get("admin1"), r.get("country")]
        label = ", ".join(p for p in label_parts if p)
        return r["latitude"], r["longitude"], label
    except Exception:
        return None


def geocode_address(address: str):
    # Normalize so "123 Main St" and "123 main st" share a cache entry.
    return _geocode_address_cached(address.strip().lower())


_current_weather_cache = {}


def fetch_current_weather(lat, lon):
    key = (_round_coord(lat), _round_coord(lon))
    cached = _current_weather_cache.get(key)
    now = time.time()
    if cached is not None and (now - cached[1]) < CURRENT_WEATHER_TTL_SECONDS:
        return cached[0]

    try:
        resp = _session.get(
            WEATHER_FORECAST_URL,
            params={"latitude": lat, "longitude": lon, "current": WEATHER_HOURLY_VARS, "timezone": "auto"},
            timeout=10,
        )
        resp.raise_for_status()
        c = resp.json().get("current", {})
        result = {
            "Temperature_C": c.get("temperature_2m", np.nan),
            "Windspeed_kmh": c.get("windspeed_10m", np.nan),
        }
        _current_weather_cache[key] = (result, now)
        return result
    except Exception:
        return None


@lru_cache(maxsize=512)
def fetch_historical_weather(lat, lon, day_str, hour):
    try:
        resp = _session.get(
            WEATHER_ARCHIVE_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": day_str, "end_date": day_str,
                "hourly": WEATHER_HOURLY_VARS, "timezone": "auto",
            },
            timeout=15,
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
    except Exception:
        hourly = {}

    times = hourly.get("time", [])
    if not times:
        return {c: np.nan for c in WEATHER_COLUMNS}
    target = f"{day_str}T{hour:02d}:00"
    idx = times.index(target) if target in times else min(hour, len(times) - 1)

    def _get(var):
        arr = hourly.get(var, [])
        return arr[idx] if idx < len(arr) else np.nan

    return {
        "Temperature_C": _get("temperature_2m"),
        "Windspeed_kmh": _get("windspeed_10m"),
    }


def _grid_cell(lat, lon, cell_size=RISK_CELL_SIZE):
    return (round(lat / cell_size), round(lon / cell_size))


def lookup_local_risk(lat, lon, risk_map):
    cell = _grid_cell(lat, lon)
    return risk_map.get(cell, risk_map.get("__global__", 0.0))


def match_category(raw_value: str, valid_values: set) -> str:
    raw_upper = raw_value.strip().upper()
    for v in valid_values:
        if v.strip().upper() == raw_upper:
            return v
    return "OTHER"


def resolve_weather(lat, lon, crash_date, hour, weather_medians):
    """
    - Past date: try historical weather for that exact date/hour. If the
      archive API doesn't have complete data, fall back to the TRAINING
      MEDIANS -- never to today's live weather, since that would silently
      mix in the wrong point in time for a historical prediction.
    - Today / future date: use live/forecast weather, since that's the
      best available signal for a "right now" or near-future prediction.
      If live weather is unavailable, fall back to training medians too.
    """
    today = datetime.now(ZoneInfo('America/New_York')).date()
    lat_r, lon_r = _round_coord(lat), _round_coord(lon)

    if crash_date is not None and crash_date < today:
        day_str = crash_date.strftime("%Y-%m-%d")
        weather = fetch_historical_weather(lat_r, lon_r, day_str, hour)
        if not any(pd.isna(v) for v in weather.values()):
            return weather, "historical"
        return {col: weather_medians.get(col, 0.0) for col in WEATHER_COLUMNS}, "fallback (typical values for this date's data being unavailable)"

    weather = fetch_current_weather(lat_r, lon_r)
    if weather is not None and not any(pd.isna(v) for v in weather.values()):
        return weather, "live"

    return {col: weather_medians.get(col, 0.0) for col in WEATHER_COLUMNS}, "fallback (typical values)"


def build_prediction_input(latitude, longitude, hour, day_of_week, month, crash_date,
                            contributing_factor_raw, vehicle_type_raw, vehicle_count,
                            feature_columns, category_maps, risk_map, weather_medians):
    validate_location(latitude, longitude)
    validate_vehicle_count(vehicle_count)

    factor_valid = category_maps.get("CONTRIBUTING FACTOR VEHICLE 1", set())
    vehicle_valid = category_maps.get("VEHICLE TYPE CODE 1", set())
    contributing_factor = match_category(contributing_factor_raw, factor_valid)
    vehicle_type = match_category(vehicle_type_raw, vehicle_valid)

    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_rush_hour = 1 if ((7 <= hour <= 9) or (16 <= hour <= 19)) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    local_risk = lookup_local_risk(latitude, longitude, risk_map)

    weather, weather_source = resolve_weather(latitude, longitude, crash_date, hour, weather_medians)

    row = {
        "LATITUDE": latitude, "LONGITUDE": longitude,
        "Hour": hour, "Day_of_Week": day_of_week, "Month": month,
        "Is_Night": is_night, "Is_Rush_Hour": is_rush_hour, "Is_Weekend": is_weekend,
        "Local_Historical_Risk": local_risk, "Vehicle_Count": vehicle_count,
    }
    row.update(weather)
    input_df = pd.DataFrame([row])

    factor_col = f"CONTRIBUTING FACTOR VEHICLE 1_{contributing_factor}"
    vehicle_col = f"VEHICLE TYPE CODE 1_{vehicle_type}"
    for col in feature_columns:
        if col.startswith("CONTRIBUTING FACTOR VEHICLE 1_"):
            input_df[col] = 1 if col == factor_col else 0
        elif col.startswith("VEHICLE TYPE CODE 1_"):
            input_df[col] = 1 if col == vehicle_col else 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    meta = {
        "local_risk_pct": round(local_risk * 100, 1),
        "weather": weather,
        "weather_source": weather_source,
        "factor_was_other": contributing_factor == "OTHER",
        "vehicle_was_other": vehicle_type == "OTHER",
    }
    return input_df[feature_columns], meta


def predict(input_df, model, threshold):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["rf"].classes_
        class1_idx = list(classes).index(1)
        prediction = 1 if probabilities[class1_idx] >= threshold else 0
        prob_by_class = {int(c): float(p) for c, p in zip(classes, probabilities)}
    else:
        prediction = int(model.predict(input_df)[0])
        prob_by_class = {0: 1.0 - prediction, 1: float(prediction)}
    return prediction, prob_by_class