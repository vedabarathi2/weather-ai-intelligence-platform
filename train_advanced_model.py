"""
Advanced weather model training for Salem.
Trains separate 1h/3h/6h temperature regressors and rain classifiers.
Uses chronological holdout testing; preserves the existing baseline model.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_PATH = Path("data/weather_history_salem.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "weather_advanced_salem.joblib"
REPORT_PATH = Path("data/weather_advanced_metrics.json")

BASE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "temperature_lag_1",
    "temperature_lag_3",
    "temperature_lag_6",
    "humidity_lag_1",
    "precipitation_lag_1",
    "wind_lag_1",
]


def make_dataset():
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing input file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = [
        "time", "temperature_2m", "relative_humidity_2m",
        "precipitation", "surface_pressure", "wind_speed_10m",
        "cloud_cover",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
          .sort_values("time")
          .drop_duplicates("time")
          .reset_index(drop=True)
    )

    # Warn about missing hourly timestamps; do not fabricate weather observations.
    gaps = df["time"].diff().dropna()
    gap_count = int((gaps != pd.Timedelta(hours=1)).sum())
    if gap_count:
        warnings.warn(
            f"Found {gap_count} non-hourly timestamp gaps. Rows around gaps are not "
            "automatically interpolated; inspect the CSV if results look unusual."
        )

    hour = df["time"].dt.hour
    day = df["time"].dt.dayofyear
    month = df["time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["temperature_lag_1"] = df["temperature_2m"].shift(1)
    df["temperature_lag_3"] = df["temperature_2m"].shift(3)
    df["temperature_lag_6"] = df["temperature_2m"].shift(6)
    df["humidity_lag_1"] = df["relative_humidity_2m"].shift(1)
    df["precipitation_lag_1"] = df["precipitation"].shift(1)
    df["wind_lag_1"] = df["wind_speed_10m"].shift(1)

    # Forecast temperature at each future horizon.
    for horizon in (1, 3, 6):
        df[f"target_temp_{horizon}h"] = df["temperature_2m"].shift(-horizon)

        # Positive class means measurable rain at least once in the next N hours.
        future_rain = [
            df["precipitation"].shift(-step)
            for step in range(1, horizon + 1)
        ]
        df[f"target_rain_{horizon}h"] = (
            pd.concat(future_rain, axis=1).max(axis=1) >= 0.1
        ).astype(float)

    target_columns = [f"target_temp_{h}h" for h in (1, 3, 6)] + [
        f"target_rain_{h}h" for h in (1, 3, 6)
    ]
    return df.dropna(subset=BASE_FEATURES + target_columns).reset_index(drop=True), gap_count


def main():
    df, gap_count = make_dataset()
    if len(df) < 1000:
        raise SystemExit(f"Only {len(df)} usable rows; need at least 1,000.")

    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    x_train = train[BASE_FEATURES]
    x_test = test[BASE_FEATURES]

    print(f"Usable rows: {len(df):,}")
    print(f"Training rows: {len(train):,}; chronological test rows: {len(test):,}")
    print(f"Non-hourly gaps detected: {gap_count}")
    print(f"Test period: {test['time'].iloc[0]} to {test['time'].iloc[-1]}")
    print("\nTEMPERATURE MODEL COMPARISON (lower MAE/RMSE is better)")

    bundles = {"city": "Salem", "features": BASE_FEATURES, "temperature": {}, "rain": {}}
    report = {
        "city": "Salem",
        "usable_rows": int(len(df)),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "non_hourly_gaps": gap_count,
        "test_start": str(test["time"].iloc[0]),
        "test_end": str(test["time"].iloc[-1]),
        "temperature": {},
        "rain": {},
    }

    for horizon in (1, 3, 6):
        target = f"target_temp_{horizon}h"
        y_train = train[target].astype(float)
        y_test = test[target].astype(float)

        candidates = {
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=0.9,
                random_state=42, n_jobs=-1
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=250, min_samples_leaf=3, max_features=0.9,
                random_state=42, n_jobs=-1
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.08, max_leaf_nodes=20,
                l2_regularization=1.0, random_state=42
            ),
        }

        results = {}
        fitted = {}
        for name, model in candidates.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            mae = float(mean_absolute_error(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            results[name] = {"mae_c": mae, "rmse_c": rmse}
            fitted[name] = model
            print(f"{horizon}h | {name:20s} MAE={mae:.3f} C  RMSE={rmse:.3f} C")

        best_name = min(results, key=lambda name: results[name]["mae_c"])
        bundles["temperature"][horizon] = {
            "model": fitted[best_name],
            "model_name": best_name,
        }
        report["temperature"][f"{horizon}h"] = {
            "selected_model": best_name,
            "candidates": results,
        }

    print("\nRAINFALL MODEL EVALUATION")
    for horizon in (1, 3, 6):
        target = f"target_rain_{horizon}h"
        y_train = train[target].astype(int)
        y_test = test[target].astype(int)

        if y_train.nunique() < 2:
            print(f"{horizon}h | Skipped: training data contains only one rain class.")
            report["rain"][f"{horizon}h"] = {"status": "skipped_one_training_class"}
            bundles["rain"][horizon] = None
            continue

        rain_model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rain_model.fit(x_train, y_train)
        pred = rain_model.predict(x_test)
        probabilities = rain_model.predict_proba(x_test)
        rain_class_index = list(rain_model.classes_).index(1)
        rain_probability = probabilities[:, rain_class_index]

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        if y_test.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, rain_probability))

        bundles["rain"][horizon] = rain_model
        report["rain"][f"{horizon}h"] = metrics
        print(
            f"{horizon}h | Accuracy={metrics['accuracy']:.3f} "
            f"Precision={metrics['precision']:.3f} "
            f"Recall={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundles, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved new model: {MODEL_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print("The existing weather_model_salem.joblib was not changed.")
    print("These are historical holdout metrics, not a guarantee of future forecast accuracy.")


if __name__ == "__main__":
    main()
