import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


# Load API token
load_dotenv()

api_key = os.getenv("5e3c5ae2-6376-4ba5-a838-817c3621c86b")

if api_key is None:
    raise ValueError("ENTSOE_API_KEY was not found in .env")


# Connect to ENTSO-E
client = EntsoePandasClient(api_key=api_key)


# Create data folder
DATA_DIR = Path("data/raw/entsoe")
DATA_DIR.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp(
    "2025-01-01",
    tz="Europe/Paris"
)

END = pd.Timestamp(
    "2026-01-01",
    tz="Europe/Paris"
)


prices = client.query_day_ahead_prices(
    "FR",
    start=START,
    end=END
)

prices.name = "day_ahead_price_eur_mwh"

prices.to_csv(
    DATA_DIR / "day_ahead_prices_fr.csv",
    index_label="datetime"
)

print(prices.head())

print(
    "\nSaved to:",
    DATA_DIR / "day_ahead_prices_fr.csv"
)