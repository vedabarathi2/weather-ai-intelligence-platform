"""Download hourly historical weather for one city from Open-Meteo Archive API."""
import argparse
import re
from pathlib import Path

import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path("data")


def get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City name, for example Chennai")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise SystemExit("End date must be on or after start date.")

    geo = get_json(
        GEOCODING_URL,
        {"name": args.city, "count": 1, "language": "en", "format": "json"},
    )
    results = geo.get("results") or []
    if not results:
        raise SystemExit(f"City not found: {args.city}")
    place = results[0]

    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "surface_pressure,wind_speed_10m,cloud_cover"
        ),
        "timezone": "auto",
    }
    payload = get_json(ARCHIVE_URL, params)
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise SystemExit("No hourly historical data was returned for this date range.")

    frame = pd.DataFrame(hourly)
    frame["city"] = place.get("name", args.city)
    frame["latitude"] = place["latitude"]
    frame["longitude"] = place["longitude"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    city_slug = re.sub(r"[^a-z0-9]+", "_", place.get("name", args.city).lower()).strip("_")
    output = DATA_DIR / f"weather_history_{city_slug}.csv"
    if output.exists():
        old = pd.read_csv(output)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["time", "latitude", "longitude"], keep="last")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} hourly rows to {output}")
    print(f"Location: {place.get('name')} | {place.get('country')}")
    print("Note: archive values are model/reanalysis-based weather data, not necessarily local station observations.")


if __name__ == "__main__":
    main()
