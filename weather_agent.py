"""Hourly multi-city weather data collector for the AI Weather Companion.

Uses Open-Meteo's documented forecast endpoint (no HTML scraping/API key).
Stores one current-weather observation per city/timestamp in SQLite.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "weather_agent.db"
LOG_PATH = LOG_DIR / "weather_agent.log"
API_URL = "https://api.open-meteo.com/v1/forecast"
LOCAL_TZ = "Asia/Kolkata"

# Approximate city-centre coordinates. You can add more cities here.
CITIES: dict[str, dict[str, Any]] = {
    "Salem": {"latitude": 11.6643, "longitude": 78.1460, "country": "India"},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707, "country": "India"},
    "Coimbatore": {"latitude": 11.0168, "longitude": 76.9558, "country": "India"},
    "Madurai": {"latitude": 9.9252, "longitude": 78.1198, "country": "India"},
    "Tiruchirappalli": {"latitude": 10.7905, "longitude": 78.7047, "country": "India"},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946, "country": "India"},
}

CURRENT_FIELDS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
    "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
    "pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    observed_at TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    temperature_2m REAL,
    relative_humidity_2m REAL,
    apparent_temperature REAL,
    is_day INTEGER,
    precipitation REAL,
    rain REAL,
    showers REAL,
    snowfall REAL,
    weather_code INTEGER,
    cloud_cover REAL,
    pressure_msl REAL,
    surface_pressure REAL,
    wind_speed_10m REAL,
    wind_direction_10m REAL,
    wind_gusts_10m REAL,
    source TEXT NOT NULL DEFAULT 'Open-Meteo',
    UNIQUE(city, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_weather_city_time
ON weather_observations(city, observed_at);
"""

COLUMNS = (
    "city,country,latitude,longitude,observed_at,fetched_at_utc,"
    "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
    "precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
    "pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,source"
).split(",")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")],
        force=True,
    )


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "AI-Weather-Companion/1.0 (learning project)"})
    return session


def validate_payload(city: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = payload.get("current")
    if not isinstance(current, dict) or not current.get("time"):
        raise ValueError(f"{city}: API response is missing current weather/time")

    # Validate ranges without rejecting valid zero values or missing optional fields.
    checks = {
        "temperature_2m": (-90, 65),
        "relative_humidity_2m": (0, 100),
        "cloud_cover": (0, 100),
        "wind_speed_10m": (0, 400),
        "precipitation": (0, 2000),
    }
    for field, (low, high) in checks.items():
        value = current.get(field)
        if value is not None and not (low <= float(value) <= high):
            raise ValueError(f"{city}: invalid {field} value: {value}")

    return current


def collect_city(session: requests.Session, conn: sqlite3.Connection, city: str) -> bool:
    if city not in CITIES:
        raise ValueError(f"Unknown city '{city}'. Available cities: {', '.join(CITIES)}")

    location = CITIES[city]
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": CURRENT_FIELDS,
        "timezone": LOCAL_TZ,
    }
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    current = validate_payload(city, payload)

    values = {
        "city": city,
        "country": location["country"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "observed_at": str(current["time"]),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Open-Meteo",
    }
    for field in COLUMNS:
        if field not in values:
            values[field] = current.get(field)

    placeholders = ",".join("?" for _ in COLUMNS)
    updates = ",".join(f"{column}=excluded.{column}" for column in COLUMNS if column not in ("city", "observed_at"))
    sql = (
        f"INSERT INTO weather_observations ({','.join(COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(city, observed_at) DO UPDATE SET {updates}"
    )
    conn.execute(sql, [values[column] for column in COLUMNS])
    conn.commit()
    logging.info(
        "%s | observed %s | %.1f C | humidity %s%% | rain %s mm",
        city,
        values["observed_at"],
        float(current.get("temperature_2m") or 0),
        current.get("relative_humidity_2m", "n/a"),
        current.get("rain", "n/a"),
    )
    return True


def run_collection(cities: list[str]) -> tuple[int, int]:
    logging.info("Starting collection for: %s", ", ".join(cities))
    success = 0
    failed = 0
    session = build_session()
    conn = connect_db()
    try:
        for city in cities:
            try:
                collect_city(session, conn, city)
                success += 1
            except (requests.RequestException, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                failed += 1
                logging.exception("Collection failed for %s: %s", city, exc)
    finally:
        conn.close()
        session.close()
    logging.info("Collection finished: %d succeeded, %d failed", success, failed)
    return success, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current weather for configured cities into SQLite.")
    parser.add_argument("--cities", nargs="+", choices=sorted(CITIES), default=list(CITIES), help="Cities to collect")
    parser.add_argument("--once", action="store_true", help="Collect once and exit (default)")
    parser.add_argument("--schedule", action="store_true", help="Run now, then repeat on a schedule")
    parser.add_argument("--interval-minutes", type=int, default=60, help="Schedule interval; minimum 15 minutes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.interval_minutes < 15:
        logging.error("For this starter agent, the interval must be at least 15 minutes.")
        return 2

    run_collection(args.cities)
    if args.schedule:
        scheduler = BlockingScheduler(timezone=ZoneInfo(LOCAL_TZ))
        scheduler.add_job(
            run_collection,
            trigger="interval",
            minutes=args.interval_minutes,
            args=[args.cities],
            id="weather_collection",
            max_instances=1,
            coalesce=True,
        )
        logging.info("Agent scheduled every %d minutes. Keep this terminal open; press Ctrl+C to stop.", args.interval_minutes)
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logging.info("Weather collection agent stopped.")
    else:
        logging.info("One-time collection complete. Run with --schedule to automate collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
