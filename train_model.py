"""Train simple next-hour temperature and rain models from weather_history.csv."""
import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES = [
    "hour",
    "month",
    "day_of_year",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="Use the same city name shown by the geocoding result")
    args = parser.parse_args()
    city_slug = re.sub(r"[^a-z0-9]+", "_", args.city.lower()).strip("_")
    data_path = DATA_DIR / f"weather_history_{city_slug}.csv"
    model_path = MODEL_DIR / f"weather_model_{city_slug}.joblib"
    if not data_path.exists():
        raise SystemExit(f"Historical data not found: {data_path}. Run collect_history.py first and use the same city name.")

    frame = pd.read_csv(data_path)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    for column in FEATURES[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hour"] = frame["time"].dt.hour
    frame["month"] = frame["time"].dt.month
    frame["day_of_year"] = frame["time"].dt.dayofyear

    # Learn the next hour's values from the weather conditions at the current hour.
    frame["temperature_next_hour"] = frame["temperature_2m"].shift(-1)
    frame["rain_next_hour"] = (frame["precipitation"].shift(-1) >= 0.1).astype("Int64")
    frame = frame.dropna(subset=FEATURES + ["temperature_next_hour", "rain_next_hour"])
    if len(frame) < 300:
        raise SystemExit(f"Only {len(frame)} usable rows. Download more historical data (ideally at least several months).")

    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    x_train, x_test = train[FEATURES], test[FEATURES]

    temperature_model = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    temperature_model.fit(x_train, train["temperature_next_hour"].astype(float))
    temperature_prediction = temperature_model.predict(x_test)
    mae = mean_absolute_error(test["temperature_next_hour"].astype(float), temperature_prediction)
    print(f"Next-hour temperature test MAE: {mae:.2f} °C")

    rain_target_train = train["rain_next_hour"].astype(int)
    rain_model = None
    if rain_target_train.nunique() > 1:
        rain_model = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
        rain_model.fit(x_train, rain_target_train)
        rain_prediction = rain_model.predict(x_test)
        accuracy = accuracy_score(test["rain_next_hour"].astype(int), rain_prediction)
        print(f"Next-hour rain/no-rain test accuracy: {accuracy:.1%}")
    else:
        print("Rain classifier skipped: training data has only one class (all rain or all no-rain).")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    latitude = float(frame["latitude"].iloc[0]) if "latitude" in frame else None
    longitude = float(frame["longitude"].iloc[0]) if "longitude" in frame else None
    joblib.dump(
        {"features": FEATURES, "temperature_model": temperature_model, "rain_model": rain_model,
         "latitude": latitude, "longitude": longitude, "city": args.city},
        model_path,
    )
    print(f"Saved trained model to {model_path}")
    print("Important: this is a learning baseline. Validate across seasons and compare against official forecasts before relying on it.")


if __name__ == "__main__":
    main()
