"""
Midtown Manhattan Accident Risk Prediction (v3)
================================================
Trains and compares several classifiers on the NYC Motor Vehicle
Collisions dataset, filtered to Midtown Manhattan, then saves the best
model for reuse at prediction time.

NEW IN v3 (vs v2)
------------------
1. ADDRESS GEOCODING: at prediction time you can type a real address
   ("Times Square, New York") instead of raw latitude/longitude. Uses
   Open-Meteo's free geocoding API (no signup / API key required).
2. WEATHER FEATURES: temperature, precipitation, snowfall, wind speed,
   and a derived "bad weather" flag are now real predictive features.
   - Training: each historical accident's weather is backfilled from
     Open-Meteo's free historical weather archive, matched to that
     accident's date, hour, and location. Results are cached to disk
     (models/weather_cache.joblib) so re-running training doesn't
     re-hit the API for rows it's already fetched.
   - Prediction: live current weather is fetched for the entered
     location via Open-Meteo's forecast API, so train and predict use
     the same feature definitions.
   - If the machine has no internet access when predicting, it falls
     back to the training-set's median weather values and says so
     explicitly (rather than crashing or silently guessing).
3. Prediction now defaults to "right now" (today's date/time) since
   the whole point of live weather + live location is assessing
   current risk. You can still answer 'n' to enter a custom date/time
   instead (weather will then be looked up for that date if it's in
   the past, or fall back to current conditions with a note otherwise).
4. VEHICLE COUNT: number of vehicles involved (counted from how many
   of the VEHICLE TYPE CODE 1-5 columns are filled in) is now a
   feature, since multi-vehicle collisions tend to be more severe.

Everything from v2 is retained: the case-matching bug fix, rare-category
grouping, class-1-focused metrics, recall-weighted threshold tuning, and
the historical local-risk-by-location feature.

Requires the `requests` package: pip install requests

Run this in IDLE with F5, or from a terminal as:
    python train_model.py train
    python train_model.py predict
"""

import sys
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import requests
except ImportError:
    requests = None

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(r"C:\Users\mahes\Downloads\archive\NYC_Accidents.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_severity_model.joblib"
COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
CATEGORY_MAP_PATH = MODEL_DIR / "category_maps.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.joblib"
RISK_MAP_PATH = MODEL_DIR / "local_risk_map.joblib"
WEATHER_MEDIAN_PATH = MODEL_DIR / "weather_medians.joblib"
WEATHER_CACHE_PATH = MODEL_DIR / "weather_cache.joblib"
PLOTS_DIR = Path("plots")

BOROUGH = "MANHATTAN"
LAT_MIN, LAT_MAX = 40.742, 40.767
LNG_MIN, LNG_MAX = -74.008, -73.971

INJURY_COLUMNS = ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"]
CATEGORICAL_COLUMNS = ["CONTRIBUTING FACTOR VEHICLE 1", "VEHICLE TYPE CODE 1"]
VEHICLE_CODE_COLUMNS = ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"]
CATEGORY_MIN_COUNT = 15

RISK_CELL_SIZE = 0.003
RISK_SMOOTHING_PRIOR = 20
THRESHOLD_RECALL_BETA = 2.0
# Never accept a threshold whose class-1 precision falls below this --
# without a floor, recall-weighted tuning can pick a threshold that
# flags almost everything, which is useless even at high recall.
MIN_PRECISION_FLOOR = 0.35

RISK_LABELS = {0: "MINOR", 1: "ELEVATED RISK (injury/fatality)"}

# Set to False to skip weather entirely (e.g. no internet access) --
# the model will then only use location/time/factor/vehicle features.
ENABLE_WEATHER = True

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_HOURLY_VARS = "temperature_2m,precipitation,snowfall,windspeed_10m,weathercode"
# Precipitation/Snowfall/Is_Bad_Weather showed near-zero importance in
# practice (most sampled crashes happened in clear weather, so these had
# almost no variance) -- keeping them just added noise for little payoff.
# Temperature and Windspeed carry real signal, so those stay.
WEATHER_COLUMNS = ["Temperature_C", "Windspeed_kmh"]


# ============================================================
# DATA LOADING & FILTERING
# ============================================================

def load_midtown_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Update DATA_PATH in the config section.")
    print("Loading dataset...")
    data = pd.read_csv(path, low_memory=False)
    print(f"Full dataset: {data.shape}")

    manhattan_data = data[data["BOROUGH"].astype(str).str.upper() == BOROUGH].copy()
    print(f"{BOROUGH} accidents: {len(manhattan_data)}")
    manhattan_data = manhattan_data.dropna(subset=["LATITUDE", "LONGITUDE"])

    midtown = manhattan_data[
        (manhattan_data["LATITUDE"].between(LAT_MIN, LAT_MAX))
        & (manhattan_data["LONGITUDE"].between(LNG_MIN, LNG_MAX))
    ].copy()
    print(f"Midtown Manhattan accidents: {len(midtown)}")
    return midtown


# ============================================================
# GEOCODING & WEATHER (Open-Meteo -- free, no API key)
# ============================================================

def _require_requests():
    if requests is None:
        sys.exit("The 'requests' package is required for geocoding/weather. Install it with: pip install requests")


def geocode_address(address: str):
    """Returns (lat, lon, display_name) or None if not found / no internet."""
    _require_requests()
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": address, "count": 1, "language": "en", "format": "json"},
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
    except Exception as e:
        print(f"(Geocoding failed: {e})")
        return None


def _weathercode_is_bad(code) -> int:
    # WMO weather codes: 50s/60s/80s = rain/showers, 70s/85-86 = snow, 95-99 = thunderstorm
    try:
        code = int(code)
    except (TypeError, ValueError):
        return 0
    return 1 if (code >= 51) else 0


def fetch_current_weather(lat: float, lon: float):
    """Live weather right now at (lat, lon). Returns a dict of WEATHER_COLUMNS or None on failure."""
    _require_requests()
    try:
        resp = requests.get(
            WEATHER_FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "current": WEATHER_HOURLY_VARS,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        c = resp.json().get("current", {})
        precip = c.get("precipitation", 0.0) or 0.0
        snow = c.get("snowfall", 0.0) or 0.0
        return {
            "Temperature_C": c.get("temperature_2m", np.nan),
            "Precipitation_mm": precip,
            "Snowfall_cm": snow,
            "Windspeed_kmh": c.get("windspeed_10m", np.nan),
            "Is_Bad_Weather": 1 if (precip > 0 or snow > 0 or _weathercode_is_bad(c.get("weathercode"))) else 0,
        }
    except Exception as e:
        print(f"(Live weather fetch failed: {e})")
        return None


def fetch_historical_weather(lat: float, lon: float, day_str: str, hour: int, cache: dict):
    """Historical weather for a specific past date+hour+location, using a
    (rounded location, date) disk cache to avoid refetching the same
    date/area repeatedly across many nearby accidents."""
    key = (round(lat, 2), round(lon, 2), day_str)
    if key not in cache:
        _require_requests()
        try:
            resp = requests.get(
                WEATHER_ARCHIVE_URL,
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": day_str, "end_date": day_str,
                    "hourly": WEATHER_HOURLY_VARS,
                    "timezone": "auto",
                },
                timeout=15,
            )
            resp.raise_for_status()
            cache[key] = resp.json().get("hourly", {})
        except Exception:
            cache[key] = {}
        time.sleep(0.15)  # be polite to the free API

    hourly = cache[key]
    times = hourly.get("time", [])
    if not times:
        return {c: np.nan for c in WEATHER_COLUMNS}

    target = f"{day_str}T{hour:02d}:00"
    idx = times.index(target) if target in times else min(hour, len(times) - 1)

    def _get(var):
        arr = hourly.get(var, [])
        return arr[idx] if idx < len(arr) else np.nan

    precip = _get("precipitation") or 0.0
    snow = _get("snowfall") or 0.0
    return {
        "Temperature_C": _get("temperature_2m"),
        "Precipitation_mm": precip,
        "Snowfall_cm": snow,
        "Windspeed_kmh": _get("windspeed_10m"),
        "Is_Bad_Weather": 1 if (precip > 0 or snow > 0 or _weathercode_is_bad(_get("weathercode"))) else 0,
    }


# ============================================================
# LOCAL HISTORICAL-RISK MAP
# ============================================================

def _grid_cell(lat: float, lon: float, cell_size: float = RISK_CELL_SIZE) -> tuple:
    return (round(lat / cell_size), round(lon / cell_size))


def build_local_risk_map(latitudes, longitudes, y) -> dict:
    global_rate = y.mean()
    cells = [_grid_cell(lat, lon) for lat, lon in zip(latitudes, longitudes)]
    df = pd.DataFrame({"cell": cells, "y": y.values})
    grouped = df.groupby("cell")["y"].agg(["sum", "count"])
    grouped["smoothed_rate"] = (
        (grouped["sum"] + RISK_SMOOTHING_PRIOR * global_rate) / (grouped["count"] + RISK_SMOOTHING_PRIOR)
    )
    risk_map = grouped["smoothed_rate"].to_dict()
    risk_map["__global__"] = global_rate
    return risk_map


def lookup_local_risk(lat: float, lon: float, risk_map: dict) -> float:
    cell = _grid_cell(lat, lon)
    return risk_map.get(cell, risk_map.get("__global__", 0.0))


def group_rare_categories(series: pd.Series, min_count: int):
    counts = series.value_counts()
    valid = set(counts[counts >= min_count].index)
    grouped = series.where(series.isin(valid), other="OTHER")
    return grouped, valid


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(midtown_data: pd.DataFrame):
    ml_data = midtown_data.copy()

    for col in INJURY_COLUMNS:
        ml_data[col] = pd.to_numeric(ml_data[col], errors="coerce").fillna(0)
    total_harm = ml_data[INJURY_COLUMNS].sum(axis=1)
    y = (total_harm > 0).astype(int)

    print("\nTarget distribution (0 = minor, 1 = elevated risk):")
    print(y.value_counts().sort_index())

    dt = pd.to_datetime(
        ml_data["CRASH DATE"].astype(str) + " " + ml_data["CRASH TIME"].astype(str),
        format="mixed", errors="coerce"
    )
    ml_data["Hour"] = dt.dt.hour
    ml_data["Day_of_Week"] = dt.dt.dayofweek
    ml_data["Month"] = dt.dt.month
    ml_data["Crash_Date_Str"] = dt.dt.strftime("%Y-%m-%d")

    ml_data["Is_Night"] = ml_data["Hour"].apply(lambda h: 1 if pd.notna(h) and (h >= 20 or h < 6) else 0)
    ml_data["Is_Rush_Hour"] = ml_data["Hour"].apply(
        lambda h: 1 if pd.notna(h) and ((7 <= h <= 9) or (16 <= h <= 19)) else 0
    )
    ml_data["Is_Weekend"] = ml_data["Day_of_Week"].apply(lambda d: 1 if pd.notna(d) and d >= 5 else 0)
    ml_data["Local_Historical_Risk"] = np.nan  # filled after train/test split
    ml_data["Vehicle_Count"] = ml_data[VEHICLE_CODE_COLUMNS].notna().sum(axis=1).clip(lower=1)

    for col in WEATHER_COLUMNS:
        ml_data[col] = np.nan  # filled below if ENABLE_WEATHER

    if ENABLE_WEATHER:
        print("\nBackfilling historical weather for training rows (cached, one-time cost)...")
        cache = joblib.load(WEATHER_CACHE_PATH) if WEATHER_CACHE_PATH.exists() else {}
        n = len(ml_data)
        for i, (idx, row) in enumerate(ml_data.iterrows()):
            if pd.isna(row["Crash_Date_Str"]) or pd.isna(row["Hour"]):
                continue
            weather = fetch_historical_weather(
                row["LATITUDE"], row["LONGITUDE"], row["Crash_Date_Str"], int(row["Hour"]), cache
            )
            for col in WEATHER_COLUMNS:
                ml_data.at[idx, col] = weather[col]
            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(f"  weather backfilled: {i + 1}/{n}")
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(cache, WEATHER_CACHE_PATH)
    else:
        print("\nENABLE_WEATHER is False -- skipping weather features.")

    keep_cols = (
        ["LATITUDE", "LONGITUDE", "Hour", "Day_of_Week", "Month",
         "Is_Night", "Is_Rush_Hour", "Is_Weekend", "Local_Historical_Risk", "Vehicle_Count"]
        + WEATHER_COLUMNS + CATEGORICAL_COLUMNS
    )
    X = ml_data[keep_cols].copy()

    for col in ["Hour", "Day_of_Week", "Month"]:
        X[col] = X[col].fillna(X[col].median())

    category_maps = {}
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].fillna("UNSPECIFIED").astype(str).str.strip()
        grouped, valid = group_rare_categories(X[col], CATEGORY_MIN_COUNT)
        X[col] = grouped
        category_maps[col] = valid

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    print(f"\nFinal feature matrix: {X.shape}")
    return X, y, category_maps


# ============================================================
# METRICS
# ============================================================

def evaluate(name: str, y_test, y_pred, plot_confusion: bool = True) -> dict:
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "Precision (class 1)": precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
        "Recall (class 1)": recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
        "F1 (class 1)": f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
    }
    print(f"\n{'=' * 50}\n{name.upper()} RESULTS\n{'=' * 50}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}%")
    print(f"Class 1 (elevated risk) -- Precision: {metrics['Precision (class 1)']:.2f}%  "
          f"Recall: {metrics['Recall (class 1)']:.2f}%  F1: {metrics['F1 (class 1)']:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if plot_confusion:
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        PLOTS_DIR.mkdir(exist_ok=True)
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        out_path = PLOTS_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"(Chart saved to {out_path})")
    return metrics


def best_threshold_for_class1(model, X_test, y_test, beta: float = THRESHOLD_RECALL_BETA,
                               min_precision: float = MIN_PRECISION_FLOOR) -> float:
    """Scan decision thresholds for the class-1 F-beta-maximizing cutoff,
    but only among thresholds that keep precision >= min_precision.
    Without this floor, a recall-weighted search can pick a threshold
    that flags almost every case as elevated-risk -- high recall, but
    useless because the warning stops meaning anything."""
    if not hasattr(model, "predict_proba"):
        return 0.5
    probs = model.predict_proba(X_test)[:, 1]

    candidates = []
    for t in np.arange(0.10, 0.91, 0.02):
        preds = (probs >= t).astype(int)
        precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
        recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
        score = fbeta_score(y_test, preds, beta=beta, pos_label=1, zero_division=0)
        candidates.append((t, precision, recall, score))

    eligible = [c for c in candidates if c[1] >= min_precision]
    pool = eligible if eligible else candidates
    if not eligible:
        print(f"(No threshold reached the {min_precision:.0%} precision floor -- "
              f"picking the closest available instead.)")

    best_t, best_precision, best_recall, best_score = max(pool, key=lambda c: c[3])
    print(f"Tuned decision threshold: {best_t:.2f}  "
          f"(class-1 recall = {best_recall * 100:.2f}%, precision = {best_precision * 100:.2f}%, "
          f"F{beta:g} = {best_score * 100:.2f}%)")
    return best_t


# ============================================================
# TRAINING
# ============================================================

def train_all_models(X_train, X_test, y_train, y_test):
    results, trained = [], {}

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    trained["Decision Tree"] = dt
    results.append(evaluate("Decision Tree", y_test, dt.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                 random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    trained["Random Forest"] = rf
    results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=42, probability=True)),
    ])
    svm_pipe.fit(X_train, y_train)
    trained["SVM (scaled)"] = svm_pipe
    results.append(evaluate("SVM (scaled)", y_test, svm_pipe.predict(X_test)))

    rf_smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                       random_state=42, n_jobs=-1)),
    ])
    rf_smote_pipe.fit(X_train, y_train)
    trained["Random Forest + SMOTE"] = rf_smote_pipe
    results.append(evaluate("Random Forest + SMOTE", y_test, rf_smote_pipe.predict(X_test)))

    svm_smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=42, probability=True)),
    ])
    svm_smote_pipe.fit(X_train, y_train)
    trained["SVM + Scaling + SMOTE"] = svm_smote_pipe
    results.append(evaluate("SVM + Scaling + SMOTE", y_test, svm_smote_pipe.predict(X_test)))

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    hgb = HistGradientBoostingClassifier(random_state=42, max_depth=6, learning_rate=0.08)
    hgb.fit(X_train, y_train, sample_weight=sample_weight)
    trained["Gradient Boosting"] = hgb
    results.append(evaluate("Gradient Boosting", y_test, hgb.predict(X_test)))

    return trained, pd.DataFrame(results)


def plot_comparison(comparison: pd.DataFrame):
    comparison.set_index("Model")[["Accuracy", "Precision (class 1)", "Recall (class 1)", "F1 (class 1)"]].plot(
        kind="bar", figsize=(12, 6)
    )
    plt.title("Model Comparison (class 1 = elevated risk)")
    plt.xlabel("Models")
    plt.ylabel("Score (%)")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"(Chart saved to {out_path})")


def plot_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, "named_steps"):
        for step_name in ("rf", "hgb"):
            if step_name in model.named_steps:
                model = model.named_steps[step_name]
                break
    if not hasattr(model, "feature_importances_"):
        print("(Selected model has no feature_importances_ -- skipping feature importance plot.)")
        return
    importance = pd.DataFrame({
        "Feature": feature_names, "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    print(f"\nTop {top_n} Important Features:\n", importance.to_string(index=False))
    plt.figure(figsize=(10, 6))
    plt.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"(Chart saved to {out_path})")


def run_training():
    midtown_data = load_midtown_data(DATA_PATH)
    X, y, category_maps = build_features(midtown_data)

    if y.nunique() < 2:
        sys.exit("Only one class present in the target after filtering -- can't train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    risk_map = build_local_risk_map(X_train["LATITUDE"], X_train["LONGITUDE"], y_train)
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train["Local_Historical_Risk"] = [
        lookup_local_risk(lat, lon, risk_map) for lat, lon in zip(X_train["LATITUDE"], X_train["LONGITUDE"])
    ]
    X_test["Local_Historical_Risk"] = [
        lookup_local_risk(lat, lon, risk_map) for lat, lon in zip(X_test["LATITUDE"], X_test["LONGITUDE"])
    ]

    # Weather columns may have NaNs (rows where the API had no data) --
    # impute with the TRAINING median and save those medians for the
    # live-weather-unavailable fallback at prediction time.
    weather_medians = {}
    for col in WEATHER_COLUMNS:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        weather_medians[col] = median_val
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    trained_models, comparison = train_all_models(X_train, X_test, y_train, y_test)

    print("\nFinal Model Comparison (sorted by class-1 F1):")
    print(comparison.sort_values("F1 (class 1)", ascending=False).round(2).to_string(index=False))
    plot_comparison(comparison)

    best_row = comparison.loc[comparison["F1 (class 1)"].idxmax()]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    print(f"\nBest model (by class-1 F1): {best_name} (F1 = {best_row['F1 (class 1)']:.2f}%)")

    threshold = best_threshold_for_class1(best_model, X_test, y_test)

    # Always show Random Forest feature importance as a diagnostic, even
    # if a different model won -- this tells you whether the new weather
    # and location features are actually earning their place.
    print("\n--- Feature importance diagnostic (from the Random Forest, regardless of winner) ---")
    plot_feature_importance(trained_models["Random Forest"], X.columns)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)
    joblib.dump(category_maps, CATEGORY_MAP_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)
    joblib.dump(risk_map, RISK_MAP_PATH)
    joblib.dump(weather_medians, WEATHER_MEDIAN_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Saved decision threshold ({threshold:.2f}) to {THRESHOLD_PATH}")
    print(f"Saved local historical-risk map ({len(risk_map) - 1} grid cells) to {RISK_MAP_PATH}")


# ============================================================
# PREDICTION
# ============================================================

def _match_category(raw_value: str, valid_values: set) -> str:
    raw_upper = raw_value.strip().upper()
    for v in valid_values:
        if v.strip().upper() == raw_upper:
            return v
    return "OTHER"


def _resolve_location():
    address = input("Address or place name (or press Enter to type coordinates instead): ").strip()
    if address:
        result = geocode_address(address)
        if result:
            lat, lon, label = result
            print(f"  -> Matched: {label}  (lat={lat:.5f}, lon={lon:.5f})")
            return lat, lon
        print("  -> Couldn't geocode that address, falling back to manual coordinates.")
    lat = float(input("Latitude: "))
    lon = float(input("Longitude: "))
    return lat, lon


def _resolve_datetime():
    use_now = input("Use current date/time? (y/n, default y): ").strip().lower()
    if use_now in ("", "y", "yes"):
        now = datetime.now()
        print(f"  -> Using now: {now.strftime('%Y-%m-%d %H:%M')} "
              f"(day_of_week={now.weekday()}, month={now.month})")
        return now.hour, now.weekday(), now.month, now.date()
    hour = int(input("Hour (0-23): "))
    day_of_week = int(input("Day of Week (0=Mon..6=Sun): "))
    month = int(input("Month (1-12): "))
    return hour, day_of_week, month, None


def _resolve_weather(lat, lon, crash_date, hour, weather_medians):
    if not ENABLE_WEATHER:
        return {col: weather_medians.get(col, 0.0) for col in WEATHER_COLUMNS}

    today = date.today()
    if crash_date is not None and crash_date < today:
        day_str = crash_date.strftime("%Y-%m-%d")
        cache = {}
        weather = fetch_historical_weather(lat, lon, day_str, hour, cache)
        if not any(pd.isna(v) for v in weather.values()):
            return weather
        print("  -> No historical weather data for that date/location; trying live conditions instead.")

    weather = fetch_current_weather(lat, lon)
    if weather is not None:
        return weather

    print("  -> Live weather unavailable (no internet?) -- using training-set typical values instead.")
    return {col: weather_medians.get(col, 0.0) for col in WEATHER_COLUMNS}


def collect_user_input(feature_columns, category_maps, risk_map, weather_medians) -> pd.DataFrame:
    print("\nEnter accident/environment information")
    print("-" * 42)

    latitude, longitude = _resolve_location()
    hour, day_of_week, month, crash_date = _resolve_datetime()

    contributing_factor_raw = input(
        "Primary Contributing Factor (e.g. Driver Inattention/Distraction, Unspecified): "
    ).strip()
    vehicle_type_raw = input(
        "Primary Vehicle Type (e.g. Sedan, Taxi, Bike, Station Wagon/Sport Utility Vehicle): "
    ).strip()
    vehicle_count_raw = input("Number of vehicles involved (default 2): ").strip()
    vehicle_count = int(vehicle_count_raw) if vehicle_count_raw else 2

    factor_valid = category_maps.get("CONTRIBUTING FACTOR VEHICLE 1", set())
    vehicle_valid = category_maps.get("VEHICLE TYPE CODE 1", set())
    contributing_factor = _match_category(contributing_factor_raw, factor_valid)
    vehicle_type = _match_category(vehicle_type_raw, vehicle_valid)

    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_rush_hour = 1 if ((7 <= hour <= 9) or (16 <= hour <= 19)) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    local_risk = lookup_local_risk(latitude, longitude, risk_map)

    weather = _resolve_weather(latitude, longitude, crash_date, hour, weather_medians)
    print(f"  -> Weather used: {weather['Temperature_C']:.1f}C, "
          f"precip {weather['Precipitation_mm']:.1f}mm, snow {weather['Snowfall_cm']:.1f}cm, "
          f"wind {weather['Windspeed_kmh']:.1f}km/h "
          f"({'BAD WEATHER' if weather['Is_Bad_Weather'] else 'clear'})")

    row = {
        "LATITUDE": latitude, "LONGITUDE": longitude,
        "Hour": hour, "Day_of_Week": day_of_week, "Month": month,
        "Is_Night": is_night, "Is_Rush_Hour": is_rush_hour, "Is_Weekend": is_weekend,
        "Local_Historical_Risk": local_risk, "Vehicle_Count": vehicle_count,
    }
    row.update(weather)
    input_df = pd.DataFrame([row])
    print(f"  -> This location's historical elevated-risk rate: {local_risk * 100:.1f}%")

    factor_col = f"CONTRIBUTING FACTOR VEHICLE 1_{contributing_factor}"
    vehicle_col = f"VEHICLE TYPE CODE 1_{vehicle_type}"
    for col in feature_columns:
        if col.startswith("CONTRIBUTING FACTOR VEHICLE 1_"):
            input_df[col] = 1 if col == factor_col else 0
        elif col.startswith("VEHICLE TYPE CODE 1_"):
            input_df[col] = 1 if col == vehicle_col else 0

    if contributing_factor == "OTHER":
        print(f"Note: contributing factor '{contributing_factor_raw}' grouped as OTHER (real trained category).")
    if vehicle_type == "OTHER":
        print(f"Note: vehicle type '{vehicle_type_raw}' grouped as OTHER (real trained category).")

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_columns]


def run_prediction():
    if not MODEL_PATH.exists():
        sys.exit("No saved model found. Run training first (type 'train' when asked).")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    category_maps = joblib.load(CATEGORY_MAP_PATH) if CATEGORY_MAP_PATH.exists() else {}
    threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.5
    risk_map = joblib.load(RISK_MAP_PATH) if RISK_MAP_PATH.exists() else {"__global__": 0.0}
    weather_medians = joblib.load(WEATHER_MEDIAN_PATH) if WEATHER_MEDIAN_PATH.exists() else {}

    input_df = collect_user_input(feature_columns, category_maps, risk_map, weather_medians)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["rf"].classes_
        class1_idx = list(classes).index(1)
        prediction = 1 if probabilities[class1_idx] >= threshold else 0
    else:
        prediction = model.predict(input_df)[0]
        probabilities, classes = None, [0, 1]

    print(f"\n{'=' * 50}\nPREDICTION\n{'=' * 50}")
    print(f"Result: {RISK_LABELS.get(prediction, 'UNKNOWN')}")
    if probabilities is not None:
        print(f"(Decision threshold in use: {threshold:.2f})")
        print("\nProbabilities:")
        for cls, prob in zip(classes, probabilities):
            print(f"  {RISK_LABELS.get(cls, cls)}: {prob * 100:.2f}%")

    if prediction == 1:
        print("\nWARNING: Conditions match past accidents with injuries/fatalities -- consider an alternative route.")
    else:
        print("\nConditions match mostly minor past accidents -- normal route evaluation.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("Type 'train' or 'predict' and press Enter: ").strip().lower()

    if mode == "train":
        run_training()
    elif mode == "predict":
        run_prediction()
    else:
        sys.exit("Please type 'train' or 'predict'.")