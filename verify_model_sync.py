"""
Run this from the same folder as train_model.py / predict_core.py
(wherever your models/ directory lives) to sanity-check that the saved
model artifacts actually match what the current code expects.

Usage:
    python verify_model_sync.py
"""
from pathlib import Path
import joblib

MODEL_DIR = Path("models")

REQUIRED_FILES = [
    "best_severity_model.joblib",
    "feature_columns.joblib",
    "category_maps.joblib",
    "decision_threshold.joblib",
    "local_risk_map.joblib",
    "weather_medians.joblib",
]

# These must match WEATHER_COLUMNS in train_model.py / predict_core.py
EXPECTED_WEATHER_COLUMNS = {"Temperature_C", "Windspeed_kmh"}

# These must match CATEGORICAL_COLUMNS in train_model.py
EXPECTED_CATEGORICAL_PREFIXES = [
    "CONTRIBUTING FACTOR VEHICLE 1_",
    "VEHICLE TYPE CODE 1_",
]

# These must exist as base (non-dummy) columns
EXPECTED_BASE_COLUMNS = {
    "LATITUDE", "LONGITUDE", "Hour", "Day_of_Week", "Month",
    "Is_Night", "Is_Rush_Hour", "Is_Weekend",
    "Local_Historical_Risk", "Vehicle_Count",
}


def main():
    print("Checking models/ directory...\n")

    missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        print("MISSING FILES:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun training first (python train_model.py train) before checking sync.")
        return
    print("All expected files are present.\n")

    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    category_maps = joblib.load(MODEL_DIR / "category_maps.joblib")
    weather_medians = joblib.load(MODEL_DIR / "weather_medians.joblib")
    model = joblib.load(MODEL_DIR / "best_severity_model.joblib")
    threshold = joblib.load(MODEL_DIR / "decision_threshold.joblib")
    risk_map = joblib.load(MODEL_DIR / "local_risk_map.joblib")

    print(f"Saved model type: {type(model).__name__}")
    print(f"Saved decision threshold: {threshold}")
    print(f"Local risk map size: {len(risk_map) - 1} grid cells\n")

    # --- Check 1: base columns present ---
    feature_set = set(feature_columns)
    missing_base = EXPECTED_BASE_COLUMNS - feature_set
    if missing_base:
        print(f"MISMATCH: feature_columns.joblib is missing base columns: {missing_base}")
        print("  -> This usually means the model was trained with an OLDER version")
        print("     of train_model.py (e.g. before Vehicle_Count or Local_Historical_Risk existed).")
    else:
        print("Base feature columns: OK (all expected columns present)")

    # --- Check 2: weather columns present ---
    weather_cols_in_features = {c for c in EXPECTED_WEATHER_COLUMNS if c in feature_set}
    if weather_cols_in_features != EXPECTED_WEATHER_COLUMNS:
        print(f"MISMATCH: expected weather columns {EXPECTED_WEATHER_COLUMNS}, "
              f"found {weather_cols_in_features} in feature_columns.joblib")
    else:
        print("Weather feature columns: OK")

    if set(weather_medians.keys()) != EXPECTED_WEATHER_COLUMNS:
        print(f"MISMATCH: weather_medians.joblib has keys {set(weather_medians.keys())}, "
              f"expected {EXPECTED_WEATHER_COLUMNS}")
    else:
        print("Weather medians: OK")

    # --- Check 3: categorical dummy columns exist for each expected prefix ---
    for prefix in EXPECTED_CATEGORICAL_PREFIXES:
        matching = [c for c in feature_columns if c.startswith(prefix)]
        if not matching:
            print(f"MISMATCH: no dummy columns found with prefix '{prefix}' -- "
                  f"category_maps.joblib / feature_columns.joblib may be out of sync "
                  f"with CATEGORICAL_COLUMNS in train_model.py")
        else:
            print(f"Categorical columns for '{prefix.strip('_')}': OK ({len(matching)} categories)")

    # --- Check 4: category_maps keys match CATEGORICAL_COLUMNS ---
    expected_cat_cols = {"CONTRIBUTING FACTOR VEHICLE 1", "VEHICLE TYPE CODE 1"}
    if set(category_maps.keys()) != expected_cat_cols:
        print(f"MISMATCH: category_maps.joblib has keys {set(category_maps.keys())}, "
              f"expected {expected_cat_cols}")
    else:
        print("category_maps.joblib keys: OK")

    # --- Check 5: model's expected input width matches feature_columns length ---
    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "n_features_in_"):
                n_features_in = step.n_features_in_
                break
    if n_features_in is not None:
        if n_features_in != len(feature_columns):
            print(f"MISMATCH: model expects {n_features_in} input features, "
                  f"but feature_columns.joblib has {len(feature_columns)} entries")
        else:
            print(f"Model input width vs feature_columns.joblib: OK ({n_features_in} features)")
    else:
        print("(Could not determine model's expected input width -- skipping that check.)")

    print("\nDone. If any MISMATCH lines printed above, re-run training "
          "(python train_model.py train) to regenerate all artifacts together, "
          "then re-run this check.")


if __name__ == "__main__":
    main()