import os  # Read your API key.
from pathlib import Path  # Handle file and folder paths.

import pandas as pd  # Handle dates and combine tables.
from dotenv import load_dotenv  # Read your .env file.
from entsoe import EntsoePandasClient  # Connect to ENTSO-E.


# 1. Load your API key and prepare the API client.
base = Path(__file__).resolve().parents[2]  # Project root (two levels above this file).
load_dotenv(base / ".env")  # Load settings from your project's .env.

client = EntsoePandasClient(
    api_key=os.environ["ENTSOE_API_KEY"],  # Use your private key.
    timeout=120,  # Allow longer waits for each request.
)


# 2. Choose where to save the results.
folder = base / "Data" / "Raw" / "ENSTOE"
monthly = folder / "balancing_monthly"  # Keep monthly downloads here.
monthly.mkdir(parents=True, exist_ok=True)  # Create missing folders.


# 3. Create month boundaries covering 2021–2024.
dates = pd.date_range(
    start="2021-01-01",
    end="2025-01-01",  # Final boundary; January 2025 is not downloaded.
    freq="MS",  # MS means the start of each month.
    tz="Europe/Paris",
)

files = []  # Record the monthly files needed for the final CSV.


# 4. Download each month separately.
for start, end in zip(dates[:-1], dates[1:]):
    # Pair consecutive boundaries: Jan–Feb, Feb–Mar, and so on.
    file = monthly / f"balancing_{start:%Y_%m}.csv"
    files.append(file)

    # Skip months already saved, allowing you to resume later.
    if file.exists():
        print(f"Already saved: {file.name}")
        continue

    print(f"Downloading {start:%Y-%m}...", flush=True)

    try:
        # Request French balancing energy prices for this month.
        data = client.query_activated_balancing_energy_prices(
            "FR",
            start=start,
            end=end,
        )

        # Exclude the next month's first interval.
        data = data.loc[(data.index >= start) & (data.index < end)]

        # Do not mark an empty response as a completed month.
        if data.empty:
            print("No data returned.")
            continue

        # Write a temporary file first, then rename after saving finishes.
        # An interrupted write will not appear to be a completed CSV.
        temporary = file.with_suffix(".tmp")
        data.to_csv(temporary, index_label="datetime")
        temporary.replace(file)

        print(f"Saved {file.name}")

    except Exception as error:
        # Report the error type without printing URLs containing your key.
        # Continue to the next month; rerunning retries unsaved months.
        print(f"Failed {start:%Y-%m}: {type(error).__name__}")


# 5. Combine the monthly files only when all months are saved.
if all(file.exists() for file in files):
    combined = pd.concat(
        [pd.read_csv(file) for file in files],  # Read each monthly CSV.
        ignore_index=True,  # Give the combined table a fresh row index.
    )

    output = folder / "balancing_energy_prices_fr.csv"

    # Replace the existing combined CSV only after writing completes.
    temporary = output.with_suffix(".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output)

    print(f"Combined CSV saved: {output.name}")

else:
    # Keep your existing combined CSV if some months failed.
    print("Some months are missing. Rerun to retry them.")