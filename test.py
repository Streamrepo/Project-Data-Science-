import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from requests.exceptions import HTTPError

load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv("ENTSOE_API_KEY")

if not api_key:
    raise SystemExit("Missing ENTSOE_API_KEY in .env")

client = EntsoePandasClient(api_key=api_key, timeout=30)

try:
    prices = client.query_day_ahead_prices(
        "FR",
        start=pd.Timestamp("2025-01-01", tz="Europe/Paris"),
        end=pd.Timestamp("2025-01-02", tz="Europe/Paris"),
    )

    if prices.empty:
        print("Request returned no prices.")
    else:
        print("Success! Your API key works.")
        print(prices.head())

except HTTPError as error:
    status = error.response.status_code if error.response is not None else "unknown"
    print(f"API request failed. HTTP status: {status}")

except Exception as error:
    print(f"Request failed: {type(error).__name__}")