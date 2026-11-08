import json
import math
import re
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODEL_DIR = Path("models")
DATABASE_PATH = Path("data/weather_agent.db")
ADVANCED_MODEL_PATH = MODEL_DIR / "weather_advanced_salem.joblib"
ADVANCED_METRICS_PATH = Path("data/weather_advanced_metrics.json")
PRESET_CITIES = ["Salem", "Chennai", "Coimbatore", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Kochi"]

st.set_page_config(page_title="AI Weather Intelligence", page_icon="🌦️", layout="wide")
st.title("🌦️ AI Weather Intelligence Platform")
st.caption("Multi-city weather monitoring · 7-day forecasts · Historical analytics · Local ML predictions")


def get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600, show_spinner=False)
def find_city(city: str) -> dict:
    payload = get_json(GEOCODING_URL, {"name": city, "count": 1, "language": "en", "format": "json"})
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"City not found: {city}")
    return results[0]


@st.cache_data(ttl=900, show_spinner=False)
def get_forecast(latitude: float, longitude: float) -> dict:
    return get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,weather_code",
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
            "forecast_days": 7,
            "timezone": "auto",
        },
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_historical_data(database_path: str, city: str, limit: int = 5000) -> pd.DataFrame:
    """Load the most recent observations for one city from the agent's SQLite database."""
    if not Path(database_path).exists():
        return pd.DataFrame()

    with sqlite3.connect(database_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='weather_observations'"
        ).fetchone()
        if table is None:
            return pd.DataFrame()

        return pd.read_sql_query(
            """
            SELECT observed_at, fetched_at_utc, city, temperature_2m,
                   relative_humidity_2m, apparent_temperature, precipitation,
                   rain, cloud_cover, pressure_msl, surface_pressure,
                   wind_speed_10m, wind_direction_10m, wind_gusts_10m, source
            FROM weather_observations
            WHERE city = ?
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            connection,
            params=(city, limit),
        )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_advanced_feature_row(current: dict, current_time: str, history_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Build the same 18 input features used by train_advanced_model.py.

    Historical lag values are matched to the expected earlier timestamps. The
    function refuses to guess when the database does not contain those readings.
    """
    timestamp = pd.to_datetime(current_time, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("The live weather timestamp is missing or invalid.")

    live_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "cloud_cover",
    ]
    row = {}
    for column in live_columns:
        value = current.get(column)
        if value is None or pd.isna(value):
            raise ValueError(f"Live weather data is missing '{column}'. Refresh the dashboard and try again.")
        row[column] = float(value)

    hour = timestamp.hour
    day = timestamp.dayofyear
    month = timestamp.month
    row.update({
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "day_sin": math.sin(2 * math.pi * day / 365.25),
        "day_cos": math.cos(2 * math.pi * day / 365.25),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    })

    if history_df.empty or "observed_at" not in history_df.columns:
        raise ValueError("Salem historical readings are not available. Start weather_agent.py and collect hourly readings first.")

    history = history_df.copy()
    history["observed_at"] = pd.to_datetime(history["observed_at"], errors="coerce")
    history = history.dropna(subset=["observed_at"]).sort_values("observed_at")
    # Do not use a database row at or after the live timestamp as a past lag.
    history = history[history["observed_at"] < timestamp]
    if history.empty:
        raise ValueError("No historical Salem readings exist before the current weather timestamp.")

    # A 20-minute tolerance accommodates small collection/API timestamp offsets,
    # while preventing old or irregular observations from being silently reused.
    lag_sources = {
        "temperature_lag_1": (1, "temperature_2m"),
        "temperature_lag_3": (3, "temperature_2m"),
        "temperature_lag_6": (6, "temperature_2m"),
        "humidity_lag_1": (1, "relative_humidity_2m"),
        "precipitation_lag_1": (1, "precipitation"),
        "wind_lag_1": (1, "wind_speed_10m"),
    }
    for feature_name, (hours_back, source_column) in lag_sources.items():
        expected_time = timestamp - pd.Timedelta(hours=hours_back)
        candidates = history[
            (history["observed_at"] - expected_time).abs() <= pd.Timedelta(minutes=20)
        ]
        if candidates.empty or source_column not in candidates.columns:
            raise ValueError(
                f"Missing historical value for {feature_name} (expected around "
                f"{expected_time:%Y-%m-%d %H:%M}). Keep the weather collector running so the required hourly history is available."
            )
        value = candidates.iloc[-1][source_column]
        if pd.isna(value):
            raise ValueError(f"Historical value '{source_column}' is empty for {expected_time:%Y-%m-%d %H:%M}.")
        row[feature_name] = float(value)

    missing_features = [name for name in features if name not in row]
    if missing_features:
        raise ValueError(f"The saved model expects unsupported input features: {missing_features}")

    sample = pd.DataFrame([{name: row[name] for name in features}], columns=features)
    return sample


def load_advanced_metrics() -> dict:
    if not ADVANCED_METRICS_PATH.exists():
        return {}
    try:
        return json.loads(ADVANCED_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


with st.sidebar:
    st.header("📍 Locations")
    selected_cities = st.multiselect("Compare cities", PRESET_CITIES, default=["Salem", "Chennai", "Coimbatore"])
    custom_city = st.text_input("Add another city (optional)", placeholder="e.g. Erode")
    if custom_city.strip() and custom_city.strip().title() not in selected_cities:
        selected_cities = selected_cities + [custom_city.strip().title()]
    if not selected_cities:
        selected_cities = ["Salem"]
    primary_city = st.selectbox("Main dashboard city", selected_cities, index=0)
    refresh = st.button("🔄 Refresh weather", width="stretch")
    if refresh:
        st.cache_data.clear()
    st.caption("Weather data: Open-Meteo. Cached briefly to reduce repeated requests.")

city_data = []
errors = []
with st.spinner("Loading weather for selected cities..."):
    for city in selected_cities:
        try:
            place = find_city(city)
            weather = get_forecast(float(place["latitude"]), float(place["longitude"]))
            city_data.append({"requested_city": city, "place": place, "weather": weather})
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{city}: {exc}")

if errors:
    for error in errors:
        st.warning(error)
if not city_data:
    st.error("No weather data could be loaded. Check your internet connection and city names.")
    st.stop()

primary = next((item for item in city_data if item["requested_city"] == primary_city), city_data[0])
place = primary["place"]
weather = primary["weather"]
current = weather.get("current", {})
units = weather.get("current_units", {})
location_text = ", ".join(part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part)

st.subheader(f"📍 {location_text}")
st.caption(f"Updated: {current.get('time', 'not available')} · Time zone: {weather.get('timezone', 'local')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Temperature", f"{current.get('temperature_2m', '—')} {units.get('temperature_2m', '°C')}")
c2.metric("💧 Humidity", f"{current.get('relative_humidity_2m', '—')} {units.get('relative_humidity_2m', '%')}")
c3.metric("💨 Wind speed", f"{current.get('wind_speed_10m', '—')} {units.get('wind_speed_10m', 'km/h')}")
c4.metric("🌧️ Precipitation", f"{current.get('precipitation', '—')} {units.get('precipitation', 'mm')}")

st.divider()
left, right = st.columns([1.5, 1])
with left:
    st.subheader("Next 24 hours")
    hourly = weather.get("hourly", {})
    if hourly.get("time"):
        hourly_df = pd.DataFrame({
            "Time": pd.to_datetime(hourly["time"]),
            "Temperature (°C)": hourly.get("temperature_2m", []),
            "Rain probability (%)": hourly.get("precipitation_probability", []),
        }).head(24)
        fig = px.line(hourly_df, x="Time", y="Temperature (°C)", markers=True, title="Hourly temperature")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Hourly forecast is not available.")
with right:
    st.subheader("7-day forecast")
    daily = weather.get("daily", {})
    if daily.get("time"):
        daily_df = pd.DataFrame({
            "Date": daily["time"],
            "Maximum (°C)": daily.get("temperature_2m_max", []),
            "Minimum (°C)": daily.get("temperature_2m_min", []),
            "Rain chance (%)": daily.get("precipitation_probability_max", []),
            "Rain (mm)": daily.get("precipitation_sum", []),
        })
        st.dataframe(daily_df, width="stretch", hide_index=True, height=330)

st.divider()
st.subheader("🌍 Multi-city comparison")
comparison_rows = []
map_rows = []
for item in city_data:
    p = item["place"]
    c = item["weather"].get("current", {})
    comparison_rows.append({
        "City": p.get("name", item["requested_city"]),
        "Temperature (°C)": c.get("temperature_2m"),
        "Humidity (%)": c.get("relative_humidity_2m"),
        "Wind (km/h)": c.get("wind_speed_10m"),
        "Rain (mm)": c.get("precipitation"),
    })
    map_rows.append({"latitude": float(p["latitude"]), "longitude": float(p["longitude"]), "City": p.get("name", item["requested_city"])})
comparison_df = pd.DataFrame(comparison_rows)
col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    if len(comparison_df) > 1:
        compare_fig = px.bar(comparison_df, x="City", y="Temperature (°C)", color="City", title="Current temperature by city")
        compare_fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(compare_fig, width="stretch")
with col_b:
    st.subheader("City map")
    st.map(pd.DataFrame(map_rows), latitude="latitude", longitude="longitude", size=80, zoom=4)
    st.caption("Map pins show the selected city locations, not radar or rainfall coverage.")

# Phase 3: historical observations saved by weather_agent.py
st.divider()
st.subheader("🗄️ Historical weather analytics")
st.caption("Historical readings are loaded from data/weather_agent.db, collected by the scheduled weather agent.")

if not DATABASE_PATH.exists():
    st.warning("Historical database not found. Run weather_agent.py --once first.")
else:
    history_city_options = [item["requested_city"] for item in city_data]
    history_city = st.selectbox(
        "Historical data city",
        history_city_options,
        index=history_city_options.index(primary_city) if primary_city in history_city_options else 0,
        key="history_city",
    )
    history_df = load_historical_data(str(DATABASE_PATH), history_city)

    if history_df.empty:
        st.info(f"No historical observations are stored for {history_city} yet. Keep the collection agent running; new readings will appear after its next collection.")
    else:
        history_df["observed_at"] = pd.to_datetime(history_df["observed_at"], errors="coerce")
        history_df = history_df.dropna(subset=["observed_at"]).sort_values("observed_at")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Stored readings", f"{len(history_df):,}")
        h2.metric("Latest temperature", f"{history_df['temperature_2m'].iloc[-1]:.1f} °C" if pd.notna(history_df["temperature_2m"].iloc[-1]) else "—")
        h3.metric("Latest humidity", f"{history_df['relative_humidity_2m'].iloc[-1]:.0f}%" if pd.notna(history_df["relative_humidity_2m"].iloc[-1]) else "—")
        h4.metric("Latest rainfall", f"{history_df['rain'].iloc[-1]:.1f} mm" if pd.notna(history_df["rain"].iloc[-1]) else "—")

        hist_left, hist_right = st.columns(2)
        with hist_left:
            if history_df["temperature_2m"].notna().any():
                temp_fig = px.line(
                    history_df, x="observed_at", y="temperature_2m", markers=True,
                    title=f"{history_city} · Historical temperature",
                    labels={"observed_at": "Observation time", "temperature_2m": "Temperature (°C)"},
                )
                temp_fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(temp_fig, width="stretch")
            else:
                st.info("No temperature history is available.")
        with hist_right:
            if history_df["rain"].notna().any():
                rain_fig = px.bar(
                    history_df, x="observed_at", y="rain",
                    title=f"{history_city} · Historical rainfall",
                    labels={"observed_at": "Observation time", "rain": "Rainfall (mm)"},
                )
                rain_fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(rain_fig, width="stretch")
            else:
                st.info("No rainfall history is available.")

        if history_df["relative_humidity_2m"].notna().any():
            humidity_fig = px.line(
                history_df, x="observed_at", y="relative_humidity_2m", markers=True,
                title=f"{history_city} · Historical humidity",
                labels={"observed_at": "Observation time", "relative_humidity_2m": "Humidity (%)"},
            )
            humidity_fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(humidity_fig, width="stretch")

        st.markdown("**Historical observation records**")
        display_columns = [
            "observed_at", "city", "temperature_2m", "relative_humidity_2m",
            "rain", "precipitation", "wind_speed_10m", "cloud_cover", "source",
        ]
        available_columns = [column for column in display_columns if column in history_df.columns]
        display_df = history_df[available_columns].rename(columns={
            "observed_at": "Observed at",
            "city": "City",
            "temperature_2m": "Temperature (°C)",
            "relative_humidity_2m": "Humidity (%)",
            "rain": "Rain (mm)",
            "precipitation": "Precipitation (mm)",
            "wind_speed_10m": "Wind (km/h)",
            "cloud_cover": "Cloud cover (%)",
            "source": "Source",
        })
        st.dataframe(display_df.sort_values("Observed at", ascending=False), width="stretch", hide_index=True)
        st.caption("The agent currently collects one reading per city per run. Historical charts become more useful as hourly records accumulate.")

# Phase 5: advanced AI predictions. This model is trained for Salem only.
st.divider()
st.subheader("🧠 Advanced AI predictions — Salem")
st.caption("Separate 1-hour, 3-hour and 6-hour models trained using Salem historical weather data.")

is_salem = primary_city.strip().casefold() == "salem"
if not is_salem:
    st.info("The advanced AI model is currently trained only for Salem. Other cities continue to use the existing dashboard and baseline ML model below.")
elif not ADVANCED_MODEL_PATH.exists():
    st.warning(f"Advanced model not found: {ADVANCED_MODEL_PATH}. Run train_advanced_model.py first.")
else:
    try:
        advanced_bundle = joblib.load(ADVANCED_MODEL_PATH)
        if advanced_bundle.get("city", "").strip().casefold() != "salem":
            raise ValueError("The advanced model file does not identify Salem as its trained city.")
        if not current.get("time"):
            raise ValueError("The live weather timestamp is unavailable.")

        # Load Salem's own history regardless of the historical-chart city selection.
        salem_history = load_historical_data(str(DATABASE_PATH), "Salem", limit=5000)
        feature_names = advanced_bundle.get("features", [])
        feature_sample = build_advanced_feature_row(current, current["time"], salem_history, feature_names)

        temperature_predictions = {}
        rain_predictions = {}
        temperature_rows = []
        rain_rows = []

        for horizon in (1, 3, 6):
            temperature_entry = advanced_bundle.get("temperature", {}).get(horizon)
            if not temperature_entry or temperature_entry.get("model") is None:
                temperature_predictions[horizon] = None
            else:
                predicted_value = float(temperature_entry["model"].predict(feature_sample)[0])
                temperature_predictions[horizon] = predicted_value
                temperature_rows.append({"Time": f"+{horizon} hours", "Temperature (°C)": predicted_value})

            rain_model = advanced_bundle.get("rain", {}).get(horizon)
            if rain_model is None:
                rain_predictions[horizon] = None
            else:
                classes = list(rain_model.classes_)
                if 1 not in classes:
                    rain_predictions[horizon] = None
                else:
                    class_index = classes.index(1)
                    probability = float(rain_model.predict_proba(feature_sample)[0][class_index]) * 100
                    rain_predictions[horizon] = probability
                    rain_rows.append({"Period": f"Next {horizon} hours", "Rain probability (%)": probability})

        st.markdown("**Temperature predictions**")
        temp_cols = st.columns(3)
        for column, horizon in zip(temp_cols, (1, 3, 6)):
            value = temperature_predictions[horizon]
            if value is None:
                column.metric(f"{horizon}-hour prediction", "Not available")
            else:
                column.metric(f"{horizon}-hour prediction", f"{value:.1f} °C")

        if temperature_rows:
            temp_chart_df = pd.DataFrame([
                {"Time": "Now", "Temperature (°C)": float(current["temperature_2m"])},
                *temperature_rows,
            ])
            temp_fig = px.line(
                temp_chart_df, x="Time", y="Temperature (°C)", markers=True,
                title="Live temperature and advanced AI predictions",
            )
            temp_fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(temp_fig, width="stretch")

        st.markdown("**Rain-event probabilities**")
        rain_cols = st.columns(3)
        for column, horizon in zip(rain_cols, (1, 3, 6)):
            value = rain_predictions[horizon]
            if value is None:
                column.metric(f"Next {horizon} hours", "Not available")
            else:
                column.metric(f"Next {horizon} hours", f"{value:.1f}%")

        if rain_rows:
            rain_fig = px.bar(
                pd.DataFrame(rain_rows), x="Period", y="Rain probability (%)",
                title="Probability of at least one measurable rain event",
                range_y=[0, 100],
            )
            rain_fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(rain_fig, width="stretch")
        st.caption("Rain probability means at least one rainfall reading of 0.1 mm or more during the selected future window; it is not an hourly rain forecast.")

        # Display the actual holdout evaluation report produced by the training script.
        metrics_report = load_advanced_metrics()
        if metrics_report:
            with st.expander("📊 Advanced model evaluation metrics", expanded=False):
                st.caption("Chronological historical holdout-test results from train_advanced_model.py. These are not guaranteed future accuracy.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Usable records", f"{metrics_report.get('usable_rows', 0):,}")
                m2.metric("Training records", f"{metrics_report.get('training_rows', 0):,}")
                m3.metric("Testing records", f"{metrics_report.get('test_rows', 0):,}")
                if metrics_report.get("test_start") and metrics_report.get("test_end"):
                    st.caption(f"Test period: {metrics_report['test_start']} to {metrics_report['test_end']} · Non-hourly gaps detected: {metrics_report.get('non_hourly_gaps', '—')}")

                temp_metric_rows = []
                for horizon in ("1h", "3h", "6h"):
                    details = metrics_report.get("temperature", {}).get(horizon, {})
                    selected = details.get("selected_model")
                    selected_metrics = details.get("candidates", {}).get(selected, {}) if selected else {}
                    if selected:
                        temp_metric_rows.append({
                            "Horizon": horizon,
                            "Selected model": selected,
                            "MAE (°C)": round(float(selected_metrics.get("mae_c", float("nan"))), 3),
                            "RMSE (°C)": round(float(selected_metrics.get("rmse_c", float("nan"))), 3),
                        })
                if temp_metric_rows:
                    st.markdown("**Temperature model performance**")
                    st.dataframe(pd.DataFrame(temp_metric_rows), width="stretch", hide_index=True)

                rain_metric_rows = []
                for horizon in ("1h", "3h", "6h"):
                    details = metrics_report.get("rain", {}).get(horizon, {})
                    if details.get("status"):
                        rain_metric_rows.append({"Horizon": horizon, "Status": details["status"]})
                    elif details:
                        rain_metric_rows.append({
                            "Horizon": horizon,
                            "Accuracy (%)": round(float(details.get("accuracy", 0)) * 100, 1),
                            "Precision (%)": round(float(details.get("precision", 0)) * 100, 1),
                            "Recall (%)": round(float(details.get("recall", 0)) * 100, 1),
                            "F1 score": round(float(details.get("f1", 0)), 3),
                            "ROC-AUC": round(float(details["roc_auc"]), 3) if "roc_auc" in details else "—",
                        })
                if rain_metric_rows:
                    st.markdown("**Rain classification performance**")
                    st.dataframe(pd.DataFrame(rain_metric_rows), width="stretch", hide_index=True)
        else:
            st.info("Evaluation report not found or unreadable. Run train_advanced_model.py to regenerate the metrics report.")

    except Exception as exc:
        st.warning(f"Advanced AI prediction could not run: {exc}")
        st.caption("Check that the advanced model and metrics files are present, and that Salem's database contains the required recent hourly readings.")

# Existing baseline local ML section is preserved for supported cities.
st.divider()
st.subheader("🤖 Local machine-learning prediction")
city_slug = slugify(place.get("name", primary_city))
model_path = MODEL_DIR / f"weather_model_{city_slug}.joblib"
if model_path.exists():
    try:
        bundle = joblib.load(model_path)
        if bundle.get("latitude") is not None and (
            abs(float(bundle["latitude"]) - float(place["latitude"])) > 0.02
            or abs(float(bundle["longitude"]) - float(place["longitude"])) > 0.02
        ):
            st.warning("Saved model coordinates do not match this city. Retrain the model for this location.")
        else:
            timestamp = pd.Timestamp(current["time"])
            row = {
                "hour": timestamp.hour, "month": timestamp.month, "day_of_year": timestamp.dayofyear,
                "temperature_2m": current.get("temperature_2m"),
                "relative_humidity_2m": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "surface_pressure": current.get("surface_pressure"),
                "wind_speed_10m": current.get("wind_speed_10m"),
                "cloud_cover": current.get("cloud_cover"),
            }
            sample = pd.DataFrame([{name: row[name] for name in bundle["features"]}])
            predicted_temp = float(bundle["temperature_model"].predict(sample)[0])
            rain_model = bundle.get("rain_model")
            p1, p2 = st.columns(2)
            p1.metric("Predicted temperature in 1 hour", f"{predicted_temp:.1f} °C")
            if rain_model is not None and 1 in list(rain_model.classes_):
                classes = list(rain_model.classes_)
                rain_probability = float(rain_model.predict_proba(sample)[0][classes.index(1)]) * 100
                p2.metric("Predicted rain chance in 1 hour", f"{rain_probability:.0f}%")
            else:
                p2.info("Rain model is not available for this city.")
            st.caption("Experimental local ML model trained on historical archive data. Validate across seasons before operational use.")
    except Exception as exc:
        st.warning(f"Could not run the saved model: {exc}")
else:
    st.info(f"No trained local ML model found for {place.get('name', primary_city)}. Train one using that city's historical data first.")

st.caption("Weather provider: Open-Meteo. Review current API terms, attribution and usage limits before public or commercial deployment.")
