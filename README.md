# AI Weather Companion — starter project

A beginner-friendly weather app with:

1. A Streamlit dashboard for current conditions and a 7-day forecast.
2. A historical-data downloader using the Open-Meteo Archive API.
3. A simple scikit-learn model for next-hour temperature and rain/no-rain prediction.

## Recommended software

- **VS Code** — write and run the project.
- **Python 3.10+** — programming language.
- **Streamlit** — easy web-app interface.
- **Open-Meteo** — weather/geocoding API source used in this demo.
- **pandas + scikit-learn** — prepare data and train the ML model.
- **GitHub** — save versions of your project.

## 1. Install and run the weather dashboard

Open this folder in VS Code. Open Terminal and run:

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- Windows Command Prompt: `.venv\\Scripts\\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Install packages and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the app. Enter a city in the sidebar.

## 2. Download historical data

Example for Chennai (choose a completed date range):

```bash
python collect_history.py --city Chennai --start 2023-01-01 --end 2025-12-31
```

The script saves hourly weather rows to a city-specific CSV such as `data/weather_history_chennai.csv`. Use one city per dataset.

## 3. Train the first ML model

```bash
python train_model.py --city Chennai
```

The model is saved to a city-specific file such as `models/weather_model_chennai.joblib`. Use the same city name for collection and training. The app only loads the model for the matching city.

## How the data flow works

```text
City entered by user
       |
       v
Geocoding API -> latitude / longitude
       |
       +----> Forecast API -> current weather + 7-day forecast -> Streamlit UI
       |
       +----> Archive API -> historical hourly CSV -> pandas -> scikit-learn
                                                       |
                                                       v
                                                saved ML model
                                                       |
                                                       v
                                         next-hour estimate in the app
```

## What is the AI agent's role?

Do not begin by scraping random websites. This project calls documented API endpoints. A later data-agent can:

- run the approved API collection script on a schedule;
- check whether a response is missing or malformed;
- standardize units and timestamps;
- log failures and avoid duplicate rows;
- refresh the model after enough new data has accumulated.

A scheduler such as Windows Task Scheduler or cron can run a current-conditions collector. Add an agent framework only when the basic pipeline works.

## Important limitations

- Open-Meteo archive values are model/reanalysis-based and are not necessarily ground-truth readings from a nearby weather station.
- This first ML model is a learning baseline, not a reliable public weather-warning system. Test it across different seasons and locations. Compare it with official/local forecasts.
- For a serious rain forecast, use verified station/radar/satellite observations where legally and technically available, and evaluate with precision, recall, and probability calibration—not accuracy alone.
- Check the data provider's current licence, attribution requirements, rate limits and commercial-use terms before deploying or monetizing the app. Those terms can change.
- Do not scrape pages behind logins, bypass access controls, or ignore a website's terms and robots rules.

## Suggested next upgrades

1. Save incoming observations in SQLite or PostgreSQL.
2. Run an hourly collector and monitor missing data.
3. Add map/location selection and weather alerts.
4. Add a stronger time-series model only after establishing a clean dataset and a baseline.
5. Deploy after checking API terms, security, and model performance.
