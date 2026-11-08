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
