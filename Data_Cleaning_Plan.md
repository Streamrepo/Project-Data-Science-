# Data cleaning and transformation plan

Written 2026-11-21 from a direct profile of `Data/Raw/ENSTOE/` (France, 1 Jan 2021 to 31 Dec 2024). Every number below was measured on the CSVs in the repo. Items marked **Assumption** or **Open question** are not facts.

## 1. The goal

Predict, with a logistic regression, when French imbalance prices differ unusually from the day-ahead (DA) price.

- `ST_short = Short imbalance price - DA price` for the same period.
- `ST_long  = Long imbalance price  - DA price` for the same period.
- `y = 1` when the value is above the 90th percentile, `y = 0` otherwise.

**Open question (the brief was cut off).** The pasted text stops at "...the same difference for the long position, but minus". Three readings are possible:

| Reading | Definition | Consequence |
|---|---|---|
| A | Two separate targets, `ST_short` and `ST_long`, each with its own threshold | Two models. Only 2 slots are in both tails (`ST_short` above p90 and `ST_long` below p10), so they are nearly different events. |
| B | `ST_short - ST_long` | This equals `Short - Long`. DA cancels out, so it is no longer "different to day-ahead". |
| C | Something else after "minus" | Needs the missing text. |

This plan works for A and B. Only step 7 changes. **Please confirm the formula.**

"90th/90th quartile range" is read here as the 90th percentile. If a two-sided band (10th and 90th) was meant, step 7 produces two labels.

## 2. Repository overview

- `Data/Raw/ENSTOE/`: 105 MB of ENTSO-E CSVs for France. That is 8 core files, 24 cross-border files (12 `physical_flows_*`, 12 `scheduled_*`) and `balancing_monthly/` (48 monthly files).
- `download_entsoe.py` and `download_balancing.py` fetch the data with `entsoe-py`. The API key is read from `.env`, which is git-ignored.
- `test.py` is only an API-key smoke test.
- `exsplanation.md` describes each CSV. It is accurate, and it warns about the timezone, resolution and blank-value issues confirmed below.
- `Description_ofwhatisdone.md` is empty.
- No cleaning, modelling or notebook code exists yet. The previous work was download and documentation.
- Branches: `main`, `Lucs-Magic-Tree` (merged into local `main` on 2026-11-21, not pushed) and `origin/Gabriel` (not merged).
- The raw data is committed to git, and `.gitignore` does not exclude it. Do not write cleaned data back into `Data/Raw/`.

## 3. What is wrong with the data

### 3.1 Target-side files (these matter most)

**`imbalance_prices_fr.csv`** (columns `Long`, `Short`, 68,448 rows)

| Problem | Evidence | Why it matters |
|---|---|---|
| Missing settlement periods | 1,726 of 70,082 expected 30-minute slots are absent (2.5%), in 479 gap runs. The longest gap is 46 slots (about 23 hours). Missing slots by year: 2021: 415, 2022: 162, 2023: 1,065, 2024: 84. They cluster in Jan-Mar 2021, Mar-May 2023 and Jul-Aug 2023. | Rows are not evenly spaced. Lags, rolling windows and "previous slot" features are wrong unless the series is reindexed. About 1,230 of 1,461 days have exactly 48 slots. The rest are daylight-saving days (46 or 50 slots) or have gaps. |
| Resolution changes on the last day | 30-minute steps until 2024-12-30 23:30 UTC, then 15-minute on 31 Dec 2024. That day has 92 rows but only 45 distinct (Long, Short) pairs. | The 15-minute rows are the 30-minute price repeated. They look like new observations but are not. |
| Repeated identical prices | 2,815 rows repeat the previous row's (Long, Short) pair. Runs of 2 are the most common, and the longest run is 8. | May be genuine (a price held across periods) or a fill. Not confirmed. Do not delete. |
| Long never above Short | 0 rows have Long > Short. 477 rows have Long = Short. Both prices are negative in the same 4,087 rows. | A usable sanity check. |
| Extreme values | Short ranges from -3,697.79 to 4,149.06. Long ranges from -4,182.43 to 3,583.78. | The 2022-04-04 spike (Short 3,137, DA 2,988) is real, so do not clip it. The 2024-12-20 value of 4,149 needs a check against the source. |

**`day_ahead_prices_fr.csv`** (column named `0`, 35,060 rows, hourly)

| Problem | Evidence | Why it matters |
|---|---|---|
| Column name is `0` | pandas default label | Rename to `da_price`. |
| 4 missing hours | The hour starting 00:00 Paris time on 31 Dec of each year (2021, 2022, 2023 and 2024 are all missing it). The pattern suggests a download-boundary bug. | 8 imbalance slots have no DA price and give `NaN` in `ST_short` and `ST_long`. |
| Coarser than the target | DA is hourly, imbalance is 30-minute (15-minute on the last day). | Each DA hour maps to 2 slots, or 4 on the last day. |
| Negative and zero prices | 567 negative and 240 zero hours. Minimum is -134.94. | Valid. Any relative or log transform must handle them. |
| Price level shifts by year | Median DA price is 78.45 (2021), 239.34 (2022), 97.70 (2023) and 58.81 (2024). | See 3.3. |

### 3.2 Other files (candidate features)

| File | Problem | Evidence |
|---|---|---|
| `imbalance_volumes_fr.csv` | Two more rows than the price file, so the timestamp sets differ. Sign convention is unverified (see `exsplanation.md`). Known only after the period. | 68,450 rows against 68,448 |
| `actual_load_fr.csv` | 3 duplicate timestamps, and the file is out of order after row 1. 15-minute rows at the end. Known only after delivery. | Duplicates at 2024-11-15 12:00, 13:00 and 14:00 UTC |
| `load_forecast_fr.csv` | Hourly, with 15-minute rows on the last day. No issue time in the file. | 35,133 rows |
| `renewables_forecast_fr.csv` | Blank values. Offshore wind is blank in 65% of rows. Some multi-hour gaps. | Solar 54, Onshore 63, Offshore 22,672 blank of 34,985 |
| `generation_actual_fr.csv` | Two header rows. 4 column names are duplicated (Fossil Hard coal, Hydro Pumped Storage, Hydro Water Reservoir and Wind Offshore each appear as both "Aggregated" and "Consumption"). Several columns are 99-100% blank. More rows than hourly would give, so resolution is mixed. Actual data, so known only after delivery. | 35,993 rows, 16 columns |
| `balancing_energy_prices_fr.csv` | Long format (4 reserve types times Up and Down). 88,752 repeated (time, direction, type) keys. Placeholder-looking prices: RR from -14,017 to 9,999, mFRR minimum -5,454. Known only after delivery. | 757,545 rows |
| `balancing_monthly/` | Identical content to the combined file. Using both double counts. | Row totals are equal (757,545) |
| `physical_flows_*` and `scheduled_*` | Resolution differs by border (hourly, 30-minute, 15-minute). Row counts differ (34,953 to 81,573). One gap of a day in `scheduled_CH_FR`. Physical flows are actuals. Scheduled flows are day-ahead. | |

### 3.3 Timestamps and the label itself

- Timestamps carry mixed `+01:00` and `+02:00` offsets. The first row is 2020-12-31 23:00 UTC, so grouping by UTC year invents a 2020 bucket. Parse as UTC and use `Europe/Paris` for calendar features.
- A global 90th-percentile threshold gives a label rate that varies a lot by year: 5.6% (2021), 15.4% (2022), 7.8% (2023), 11.0% (2024), measured for `ST_short`. A single threshold mostly flags the 2022 crisis. The model would partly learn "is it 2022".
- The threshold must come from the training period only. A quantile over all data puts test-period information into the label.

## 4. Cleaning steps, in order

Each step is a function in a `src/clean.py`. Read from `Data/Raw/`, write to `Data/Clean/` (add it to `.gitignore`).

1. **Freeze the raw data.** Never edit `Data/Raw/`. Record the file hashes.
2. **Parse timestamps.** `pd.to_datetime(..., utc=True)`. Keep a UTC key and add a `Europe/Paris` view for hour, weekday, month and holidays.
3. **Normalise each file.** One header row, snake_case names (`0` becomes `da_price`). Sort by time. Drop exact duplicate timestamps (`actual_load`). Flatten the generation header to `source__type` so duplicates become distinct columns.
4. **Fix the resolution.** Set the base grid to **30 minutes in UTC**, 2021-01-01 00:00 Paris to 2024-12-30 23:30 UTC. **Assumption:** drop 31 Dec 2024. The alternative is to average each pair of 15-minute rows into one 30-minute row. Both are safe because the 15-minute prices are repeats.
5. **Reindex to the full grid.** Create every expected slot. Add `is_missing` and do not forward-fill the target. Drop target slots with no `Long` or `Short` from the training and test sets, but keep them in the frame so lags remain correct.
6. **Align DA to the slots.** Each slot gets the DA price of its own hour (`floor('1h')`). The 4 missing DA hours leave 8 slots with `NaN`. Drop them, or fill from the neighbouring hours and flag them. Dropping is safer because these slots define the label.
7. **Build the target.** Compute `ST_short = Short - da_price` and `ST_long = Long - da_price` (or the confirmed formula). Then:
   - Split by time first, for example train 2021-2022, validation 2023, test 2024.
   - Compute the 90th percentile on **train only**. Apply it unchanged to validation and test.
   - `y = (ST > threshold).astype(int)`.
   - **Decision to take:** use a fixed euro threshold from train, or a rolling threshold (for example the trailing 365-day 90th percentile, shifted by one day). The fixed threshold is simple but inherits the label-rate drift in 3.3. The rolling one keeps a steadier rate but defines "unusual" relative to recent history. Take a deliberate decision and write it down.
8. **Outliers.** Do not clip the raw prices or the label. The spikes are the events being predicted. For features, use a signed log (`sign(x) * log1p(abs(x))`), because prices can be negative. Treat balancing-price extremes of about 9,999 as suspect placeholders and check them against the source before trusting them.
9. **Features, and what each is known at decision time.** The exported CSVs do not store publication or issue times, so confirm these timings yourselves. Define a decision time (**Assumption:** around 13:00 Paris on D-1, after the DA auction) and use only:

   | Allowed for slots on day D | Not allowed (known only after the period) |
   |---|---|
   | DA price, load forecast, wind/solar forecast, scheduled (day-ahead) exchanges, calendar variables | `actual_load`, `generation_actual`, `physical_flows`, `imbalance_volumes`, `balancing_energy_prices` and the imbalance prices themselves |

   Lagged versions of the "not allowed" series are usable when the lag clears the decision time (for example, same slot on D-2 or earlier). Derived features: forecast load minus forecast renewables, DA price level and its change versus the previous day, net scheduled import, hour-of-day, weekday, month, holiday flag.
10. **Missing values in features.** Blank is not zero (`exsplanation.md`). Offshore wind is blank before the farms operated: confirm the first non-blank date, then fill earlier blanks with 0 and add a flag. Forward-fill other gaps for at most 2 hours. Leave longer gaps as `NaN` and drop those rows or flag them.
11. **Scaling.** Standardise numeric features using train statistics only (needed for logistic regression to converge cleanly). One-hot encode hour, weekday and month.
12. **Modelling notes.** The label is about 10% positive by construction. Use `class_weight` or a tuned decision threshold, and judge with PR-AUC and recall at fixed precision, not accuracy. Adjacent slots are strongly correlated, so use time-ordered cross-validation. Do not shuffle.

## 5. Checks to assert after cleaning

- The time index is unique, sorted and has a constant 30-minute step.
- Row count equals the expected slot count minus recorded drops.
- `Long <= Short` on every kept row.
- No `NaN` in `y` or in any feature used by the model.
- The label rate per year is printed and reviewed.
- The train maximum timestamp is earlier than the validation minimum, and the validation maximum is earlier than the test minimum.
- No feature column is derived from a series in the "not allowed" column without a safe lag.

## 6. Open questions for the team

1. The exact formula after "but minus" (section 1).
2. Percentile or two-sided band, and fixed or rolling threshold (step 7).
3. Drop 31 Dec 2024 or average its 15-minute rows (step 4).
4. The decision time that defines which features are legal (step 9).
5. Whether to download 2025 data. `download_entsoe.py` notes that `end` can be moved to 2026-01-01. The imbalance market moved to 15-minute settlement, so 2025 data would change the grid.
6. Whether the missing 2023 imbalance periods can be re-downloaded or are missing at the source.
