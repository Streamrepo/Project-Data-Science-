import os
import pandas as pd

from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

api_key = os.getenv("ENTSOE_API_KEY")

client = EntsoePandasClient(api_key=api_key)
START = pd.Timestamp("2021-01-01", tz="Europe/Paris")
END   = pd.Timestamp("2026-01-01", tz="Europe/Paris")

COUNTRY = "FR"
test_start = pd.Timestamp("2025-01-01", tz="Europe/Paris")
test_end   = pd.Timestamp("2025-01-08", tz="Europe/Paris")

prices_test = client.query_day_ahead_prices(
    "FR",
    start=test_start,
    end=test_end
)

print(prices_test.head())




