import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


# ============================================================
# 1. FIND PROJECT DIRECTORY
# ============================================================

# Works whether running from VS Code or a normal Python script
BASE_DIR = Path.cwd()


# ============================================================
# 2. LOAD API TOKEN FROM .env
# ============================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("ENTSOE_API_KEY")

if not api_key:
    raise ValueError(
        "ENTSOE_API_KEY was not found.\n"
        "Make sure your .env file contains:\n"
        "ENTSOE_API_KEY=your_token_here"
    )


# ============================================================
# 3. CONNECT TO ENTSO-E
# ============================================================

client = EntsoePandasClient(api_key=api_key)


# ============================================================
# 4. CREATE DATA DIRECTORY
# ============================================================

DATA_DIR = BASE_DIR / "data" / "raw" / "entsoe"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Data will be saved in:")
print(DATA_DIR)


# ============================================================
# 5. DATE RANGE
# ============================================================

START = pd.Timestamp(
    "2025-01-01",
    tz="Europe/Paris"
)

END = pd.Timestamp(
    "2026-01-01",
    tz="Europe/Paris"
)

COUNTRY = "FR"


# ============================================================
# 6. DAY-AHEAD ELECTRICITY PRICES
# ============================================================

print("\nDownloading day-ahead prices...")

prices = client.query_day_ahead_prices(
    COUNTRY,
    start=START,
    end=END
)

prices.name = "day_ahead_price_eur_mwh"

prices.to_csv(
    DATA_DIR / "day_ahead_prices_fr.csv",
    index_label="datetime"
)

print("Saved day-ahead prices.")
print(prices.head())


# ============================================================
# 7. ACTUAL ELECTRICITY LOAD
# ============================================================

print("\nDownloading actual load...")

actual_load = client.query_load(
    COUNTRY,
    start=START,
    end=END
)

actual_load.to_csv(
    DATA_DIR / "actual_load_fr.csv",
    index_label="datetime"
)

print("Saved actual load.")
print(actual_load.head())


# ============================================================
# 8. DAY-AHEAD LOAD FORECAST
# ============================================================

print("\nDownloading load forecast...")

load_forecast = client.query_load_forecast(
    COUNTRY,
    start=START,
    end=END
)

load_forecast.to_csv(
    DATA_DIR / "load_forecast_fr.csv",
    index_label="datetime"
)

print("Saved load forecast.")
print(load_forecast.head())


# ============================================================
# 9. ACTUAL GENERATION BY PRODUCTION TYPE
# ============================================================

print("\nDownloading generation data...")

generation = client.query_generation(
    COUNTRY,
    start=START,
    end=END
)

generation.to_csv(
    DATA_DIR / "generation_actual_fr.csv",
    index_label="datetime"
)

print("Saved generation data.")
print(generation.head())

print("\nGeneration columns:")
print(generation.columns)


# ============================================================
# 10. DAY-AHEAD WIND AND SOLAR FORECASTS
# ============================================================

print("\nDownloading wind and solar forecasts...")

renewables_forecast = client.query_wind_and_solar_forecast(
    COUNTRY,
    start=START,
    end=END
)

renewables_forecast.to_csv(
    DATA_DIR / "renewables_forecast_fr.csv",
    index_label="datetime"
)

print("Saved renewable forecasts.")
print(renewables_forecast.head())

print("\nRenewable forecast columns:")
print(renewables_forecast.columns)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n======================================")
print("ALL ENTSO-E DATA DOWNLOADED")
print("======================================")

print("\nFiles created:")

for file in DATA_DIR.glob("*.csv"):
    print(file.name)

print("\nDataset sizes:")

print("Prices:", prices.shape)
print("Actual load:", actual_load.shape)
print("Load forecast:", load_forecast.shape)
print("Generation:", generation.shape)
print("Renewables forecast:", renewables_forecast.shape)






##########





