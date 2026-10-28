import os  # Read the API key from the environment.
from pathlib import Path  # Build file and folder paths.
import pandas as pd  # Create dates with a timezone.
from dotenv import load_dotenv  # Load settings from .env.
from entsoe import EntsoePandasClient  # Request ENTSO-E data as pandas tables.

base = Path(__file__).resolve().parent  # Locate this script's folder.
load_dotenv(base / ".env")  # Read your project's .env file.
client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"], timeout=60)  # Connect using your key.
start = pd.Timestamp("2021-01-01", tz="Europe/Paris")  # First date to download.
end = pd.Timestamp("2025-01-01", tz="Europe/Paris")  # End date, excluded; change to 2026-01-01 for all of 2025.
folder = base / "Data" / "Raw" / "ENSTOE"  # Choose your existing data folder.
folder.mkdir(parents=True, exist_ok=True)  # Create the folder if needed.


def save(name, query, *areas, **options):  # Download and save one dataset.
    print(f"Downloading {name}...", flush=True)  # Show which request is running.
    try:  # Allow other downloads to continue if this one fails.
        data = query(*areas, start=start, end=end, **options)  # Request data for the dates and areas.
        data = data.loc[(data.index >= start) & (data.index < end)]  # Keep the requested date range.
        if data.empty:  # Check whether there are any results.
            print(f"No data: {name}")  # Report an empty dataset.
            return  # Skip saving an empty file.
        data.to_csv(folder / f"{name}.csv", index_label="datetime")  # Save timestamps and values as CSV.
        print(f"Saved {name}.csv")  # Confirm the file was saved.
    except Exception as error:  # Handle failed requests without stopping the remaining jobs.
        print(f"Failed {name}: {type(error).__name__}")  # Show the error type without printing your secret key.


save("day_ahead_prices_fr", client.query_day_ahead_prices, "FR")  # Day-ahead electricity prices.
save("imbalance_prices_fr", client.query_imbalance_prices, "FR")  # Imbalance settlement prices.
save("load_forecast_fr", client.query_load_forecast, "FR")  # Day-ahead electricity demand forecast.
save("actual_load_fr", client.query_load, "FR")  # Actual electricity demand.
save("generation_actual_fr", client.query_generation, "FR")  # Actual generation by energy source.
save("renewables_forecast_fr", client.query_wind_and_solar_forecast, "FR")  # Wind and solar forecasts.
save("imbalance_volumes_fr", client.query_imbalance_volumes, "FR")  # Total imbalance volumes.
save("balancing_energy_prices_fr", client.query_activated_balancing_energy_prices, "FR")  # Activated balancing energy prices.

for neighbour in ("BE", "DE_LU", "CH", "IT_NORD", "ES", "GB"):  # Neighbouring bidding zones.
    for source, target in (("FR", neighbour), (neighbour, "FR")):  # Request both export and import directions.
        save(f"scheduled_{source}_{target}", client.query_scheduled_exchanges, source, target, dayahead=True)  # Day-ahead commercial schedules.
        save(f"physical_flows_{source}_{target}", client.query_crossborder_flows, source, target)  # Actual physical flows.
