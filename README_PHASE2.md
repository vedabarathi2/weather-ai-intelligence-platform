# Phase 2 — Automated Weather Data Collection Agent

This add-on runs separately from Streamlit. It fetches documented current-weather API observations for configured cities, validates basic value ranges, and saves them in `data/weather_agent.db` (SQLite). It also writes logs to `logs/weather_agent.log`.

## Why this is an agent

This is a reliable scheduled data-collection worker: fetch → validate → store → log. It is not an LLM agent and it does not make forecasts. The existing ML model remains a separate prediction component. Later phases can add quality reports, retraining decisions, and notifications.

## Install

Extract/copy `weather_agent.py`, `requirements_agent.txt`, and this README into the root of your existing `weather_ai_starter` project, beside `app.py`.

From PowerShell in that project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_agent.txt
```

## Test one collection cycle first

```powershell
.\.venv\Scripts\python.exe weather_agent.py --once
```

This collects Salem, Chennai, Coimbatore, Madurai, Tiruchirappalli, and Bengaluru once. To collect only Salem:

```powershell
.\.venv\Scripts\python.exe weather_agent.py --cities Salem --once
```

## Start automatic hourly collection

```powershell
.\.venv\Scripts\python.exe weather_agent.py --schedule --interval-minutes 60
```

Keep this terminal open while the agent runs. Press `Ctrl+C` to stop. Run Streamlit in a separate terminal. The first collection happens immediately; later cycles run on the chosen interval. Minimum allowed interval is 15 minutes.

## Data and logs

- SQLite database: `data/weather_agent.db`
- Table: `weather_observations`
- Log file: `logs/weather_agent.log`
- Uniqueness: one row per city and API observation timestamp; rerunning the same timestamp updates rather than duplicating it.

## Add another city

Edit the `CITIES` dictionary in `weather_agent.py` and add its approximate latitude/longitude and country. Use valid coordinates.

## Responsible use

The source is Open-Meteo's documented Forecast API. It returns model-based current conditions, not necessarily a nearby station observation. Check current Open-Meteo terms, attribution requirements, and usage limits before public or commercial deployment. Do not treat this educational collector or the project's ML output as an official warning system.
