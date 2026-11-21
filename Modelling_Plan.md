# Modelling plan: leakage-safe baseline for French imbalance-settlement stress

**Source:** `French_Electricity_Project_Modelling_Brief (1).pdf` (8 pages). Sections 1 to 9 follow the brief. Section 10 is my addition from profiling the raw data, and section 11 lists decisions the brief leaves open. Both are labelled as not coming from the brief.

**Status (2026-11-21): data processing (sections 3 to 8) is done by `src/build_dataset.py` and written to `Data/Processed/`; see section 12. No model has been fitted. Waiting for Vivek's instruction before the logistic regressions or anything later.**

**Scope:** ENTSO-E France, 1 January 2021 to 31 December 2024.
**Objective:** build a reproducible, leakage-safe processed dataset, then fit interpretable baseline models for short-side and long-side settlement stress.
**Explicitly out of scope for now:** the HMM. Do not implement it until the baseline pipeline and the two logistic regressions are correct and evaluated.

## 1. Research question

Using only information available **after the French day-ahead market has cleared but before delivery**, can observable market fundamentals identify intervals of unusually severe imbalance-settlement stress for short positions and for long positions?

## 2. Quantities and targets

For each delivery interval `t`:

| Symbol | Meaning |
|---|---|
| `P_DA(t)` | French day-ahead price |
| `P_Short(t)` | imbalance settlement price, short position |
| `P_Long(t)` | imbalance settlement price, long position |

1. Signed deviations: `D_Short = P_Short - P_DA`, `D_Long = P_Long - P_DA`.
2. Direction-aligned stress: `S_Short = P_Short - P_DA`, **`S_Long = P_DA - P_Long`** (the sign is flipped for long). `S > 0` means the imbalance outcome was worse than day-ahead for that position.
3. **No absolute values** in the main stress target, because they remove the direction.
4. Two separate binary targets from the upper tail of each stress:
   - `Y_Short = 1 if S_Short > q90_Short else 0`
   - `Y_Long  = 1 if S_Long  > q90_Long  else 0`
   - `q90_Short` and `q90_Long` are estimated on the **training sample only** and then kept fixed.
5. Keep the continuous `S_Short` and `S_Long` in the processed data. Binary targets lose severity (EUR 81 and EUR 400 are both `Y = 1`), and XGBoost does not fix that. A later secondary analysis uses `ExcessStress = S - q90` conditional on `Y = 1`.

## 3. Folder layout

Never overwrite the raw CSVs.

```
Data/
  Raw/ENSTOE/        (untouched)
  Processed/         (everything produced by the pipeline)
```

The processed table must be fully reproducible from the untouched raw files by running the pipeline code.

## 4. Data preparation steps (brief section 5)

1. **Standardise timestamps.** Parse every `datetime` as timezone-aware. Use UTC internally for joins. Keep Europe/Paris time for hour, weekday, month and other calendar features. Never strip the timezone before aligning.
2. **Inspect frequency by file and year.** Compute timestamp differences in code. Do not assume a whole file is hourly, 30-minute or 15-minute.
3. **Build a 30-minute master grid for 2021-2024**, one row per 30-minute delivery interval. Hourly variables that apply to a whole hour map to both of its half-hours.
4. **Aggregate high-frequency data correctly.**
   - 15-minute MW power: average the two quarters into a half-hour.
   - MWh energy: sum.
   - Prices: time-weighted aggregation if aggregation is needed.
5. **Construct target prices and stress.** Align `P_DA`, `P_Long` and `P_Short` to the grid, then compute `D_Short`, `D_Long`, `S_Short`, `S_Long`.
6. **Construct only pre-delivery predictors for Model 1** (section 5). Keep realised same-period variables out of `X`.
7. **Create a separate explanatory dataset** holding actual load, actual generation, physical flows, imbalance volume and balancing prices. It is for ex-post interpretation, not same-period prediction. Examples: `LoadForecastError = ActualLoad - LoadForecast`, `WindForecastError = ActualWind - WindForecast`.
8. **Audit missing values.** A blank does not mean zero. Produce missingness summaries by variable and by year. Never backfill using future observations. Fit any imputer on training data only.

## 5. Predictors

**Model 1 predictors (all known after the day-ahead auction, before delivery):**
`day_ahead_price`, `load_forecast`, `wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`, `scheduled_imports`, `scheduled_exports`, `net_scheduled_imports`, cyclical hour, `day_of_week`, `month` or season.

**Scheduled-trade features:**
- `scheduled_imports = sum(all neighbour -> France day-ahead schedules)`
- `scheduled_exports = sum(all France -> neighbour day-ahead schedules)`
- `net_scheduled_imports = scheduled_imports - scheduled_exports`

The neighbours are BE, DE_LU, CH, IT_NORD, ES and GB (12 `scheduled_*` files).

**Optional:** `residual_load_forecast = load_forecast - wind_forecast - solar_forecast`. If an unregularised logistic regression contains load, wind, solar and their exact linear combination at once, inspect multicollinearity carefully. `net_scheduled_imports` is likewise an exact combination of imports and exports.

**Kept out of the first predictive model** (same-period realised values): `actual_load`, `actual_generation`, `physical_flows`, `imbalance_volume`, `balancing_energy_prices`, and the same-period imbalance prices themselves. The predictive model must receive only predictor columns. Prices, `D`, `S` and `Y` are outcomes or target-construction variables and must never be inputs.

## 6. Splits (chronological, no shuffling)

| Period | Purpose |
|---|---|
| 2021-2022 | Training |
| 2023 | Validation and model selection |
| 2024 | Internal out-of-sample test |
| 2025 (download later) | External holdout, used once after the design is fixed |

Compute `q90_Short` and `q90_Long` from the training sample only. Keep them fixed when evaluating 2023 and 2024. After the whole design is final, refit the selected model on 2021-2024 and test once on 2025.

## 7. Scaling and calendar encoding

- Standardise continuous predictors with training-set means and standard deviations only: `z = (x - mean_train) / sd_train`.
- Hour of day: `HourSin = sin(2*pi*hour/24)` and `HourCos = cos(2*pi*hour/24)`.

## 8. Expected processed modelling table

Columns: `datetime`, `datetime_paris`, `day_ahead_price`, `load_forecast`, `wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`, `scheduled_imports`, `scheduled_exports`, `net_scheduled_imports`, `hour_sin`, `hour_cos`, `day_of_week`, `month`, `D_short`, `D_long`, `S_short`, `S_long`, `Y_short`, `Y_long`, `split`.

## 9. Pipeline order and model stages

**Pipeline:** raw CSVs, then timestamp and frequency inspection, timezone standardisation, 30-minute grid, day-ahead predictor table, target-price table, `S_Short` and `S_Long`, training-only q90 thresholds, `Y_Short` and `Y_Long`, chronological split, scaled leakage-safe predictor matrix, first logistic regressions.

**Model stages (build in this order so each gain can be identified):**

| Model | Learner | Fundamentals | HMM | When |
|---|---|---|---|---|
| 1 | Logistic regression, one for `Y_Short` and one for `Y_Long` | Yes | No | **First, and the only stage in the current instruction** |
| 2 | Logistic regression + HMM regime probabilities | Yes | Yes | Only after Model 1 is stable, with every other input identical |
| 3 | XGBoost, same feature set | Yes | No | Later |
| 4 | XGBoost + HMM | Yes | Yes | Later, compared against Model 3 |

**Definition of done for the first phase (brief's final instruction):** the full baseline data pipeline is correct and validated, the two separate logistic regressions are fitted and evaluated, the table is reproducible from untouched raw files, splits are chronological, and no same-period realised information is in `X`.

## 10. Data issues the steps must handle (my addition, measured on the raw files)

These are not in the brief. They come from profiling `Data/Raw/ENSTOE/`.

- **Missing imbalance periods:** 1,726 of 70,082 expected 30-minute slots are absent. By year: 2021: 415, 2022: 162, 2023: 1,065, 2024: 84.
- **Resolution:** imbalance prices are 30-minute through 30 Dec 2024. Only 31 Dec 2024 has 15-minute rows, and those repeat the 30-minute price. Day-ahead prices are hourly throughout.
- **Missing day-ahead hours:** the hour starting 00:00 Paris time on 31 Dec is absent in every year (4 hours), which leaves 8 slots without `S`.
- **Mixed resolution inside single files:** several scheduled and physical-flow files switch between hourly, 30-minute and 15-minute during 2024 (for example `scheduled_DE_LU_FR` and `scheduled_BE_FR`), so frequency must be handled per timestamp.
- **Offshore wind forecast:** blank until 2023-08-06, so the whole training period has no offshore values.
- **Smaller issues:** three duplicate timestamps in `actual_load_fr.csv`; `generation_actual_fr.csv` has two header rows and duplicate column names; `balancing_energy_prices_fr.csv` has repeated keys; the monthly balancing files duplicate the combined file.
- **Class rates:** with a global training-only q90 the positive rate differs by period, so report the rate per split.

## 11. Decisions the brief leaves open (need Vivek's input before starting)

1. **Gaps in forecast predictors:** drop those rows, or carry the last known forecast forward for a short, fixed limit? The brief only forbids backfilling and requires training-only imputers.
2. **Offshore wind before 2023-08-06:** the brief says a blank is not zero. The farm is not reporting yet, so zero is physically plausible. Either choice should be documented. With zeros throughout the training period the standardised coefficient cannot be learned.
3. **A missing border in the scheduled sums:** treat the total as missing, or as the sum of the borders that are present?
4. **Evaluation metrics** are not specified. Suggestion: ROC-AUC, average precision (PR-AUC), Brier score and log-loss, each reported against the base rate for the split, plus precision among the top-scoring 10% of intervals.
5. **Regularisation and class weights** are not specified. Suggestion: L2 penalty with strength chosen on 2023 only, and unweighted as the primary model.
6. **Collinear predictors:** keep `net_scheduled_imports` and `residual_load_forecast` in the table but leave them out of the first fitted matrix, or include them with regularisation?
7. **2025 data:** downloading it needs the API key and a changed `end` date in `download_entsoe.py`. The brief says to do this later.

## 12. Result of the data-processing stage (2026-11-21)

*Superseded in part by section 16: the thresholds, the split table and the design-matrix row counts below were recomputed on the clean data.*

Run with `.venv/bin/python src/build_dataset.py` (about 5 seconds, reproducible: two runs give identical files, raw files untouched).

| Output in `Data/Processed/` | Content |
|---|---|
| `modelling_table.csv` | 70,128 half-hour rows (2021-2024), the section 8 columns plus `price_long`, `price_short`, `residual_load_forecast` |
| `design_matrix.csv` | scaled predictors and `Y_short`, `Y_long` for the usable rows only, scaled with training-only means and SDs |
| `scaler.json` | training means and SDs; marks `wind_offshore_forecast` as constant in training |
| `thresholds.json` | `q90_short = 77.89`, `q90_long = 78.01` (training sample only) |
| `expost_table.csv` | realised variables and forecast errors for explanation only |
| `missingness_by_year.csv`, `frequency_by_file_year.csv` | the two audits from steps 2 and 8 |

| Split | Slots | Missing target | Missing predictor | Usable | Share `Y_short = 1` | Share `Y_long = 1` |
|---|---|---|---|---|---|---|
| train (2021-22) | 35,040 | 580 | 354 | 34,111 | 10.0% | 10.0% |
| validation (2023) | 17,520 | 1,067 | 46 | 16,409 | 7.2% | 5.3% |
| test (2024) | 17,568 | 87 | 2 | 17,481 | 10.4% | 8.1% |

**Defaults applied for the section 11 decisions (not confirmed, change on request):**
1. Forecast gaps (load, wind onshore, solar, scheduled flows): the last known value is carried forward for at most 4 slots (2 hours), never backward. Longer gaps stay missing and the row is excluded.
2. Offshore wind before its first reported value (2023-08-06) is set to 0. It is constant in the training period, so it is left out of `design_matrix.csv` and cannot be learned.
3. A scheduled import or export total is missing if any of the 6 borders is missing.
4. `net_scheduled_imports` and `residual_load_forecast` are in the design matrix as scaled columns. Choosing which of the collinear columns to fit is left for the modelling stage.
5. Hour is encoded from the half-hour fraction (for example 10:30 is 10.5), so `hour_sin` and `hour_cos` distinguish the two half-hours. Day of week and season are one-hot (Monday and winter are the references).

**Checks that pass:** grid is exactly 30-minute with 1,461 days times 48 rows; `Long <= Short` on every row; splits are chronological; hourly day-ahead prices map to both half-hours; the 15-minute rows on 31 Dec 2024 are averaged; `S_Long = P_DA - P_Long`; thresholds come from training rows only and are applied unchanged to validation and test; the training z-scores have mean 0 and SD 1; the design matrix has no missing values and no price, `D`, `S` or `Y_` inputs.

**Correction made after an independent audit (2026-11-21).** The first version of the pipeline filled a missing 30-minute imbalance slot with the previous slot's price, because a single missing slot leaves a 60-minute gap and the code read that as an hourly series. It invented 315 target prices (for example 2021-01-07 13:30). The resolution rule in `covered_cells` was changed to use the finest spacing among the previous 48 gaps, so a gap is never filled. The tables were rebuilt: the thresholds, the split table above and the design matrix (68,001 usable rows) all changed slightly. The version pushed before this correction (commit `d550f00`) contains the invented prices.

**Audit result.** Every column of `modelling_table.csv` was rebuilt from the raw CSVs by a different method (resolution decided from the minute offsets present, not from gaps) and compared on all 70,128 rows. Day-ahead, long and short prices, load and renewable forecasts, scheduled imports and exports, actual load, imbalance volume and the main generation columns match with no invented and no lost values. The only slots not compared are those within 3 hours of a file's single resolution switch (at most 26 slots per column) and two isolated slots that were checked by hand. Carried-forward cells (at most 4 slots) equal the last real value. Targets, thresholds, calendar columns, the split rule, the scaler and the design matrix were also recomputed and match.

## 13. Full audit of the processed tables (2026-11-21)

Method: every column was rebuilt from the raw CSVs by methods different from the pipeline's, and compared slot by slot on the whole 70,128-slot grid. The dangerous direction is a value in the table that the raw data cannot support ("invented"). The harmless direction is a slot left blank although a value could have been derived ("conservatively blank").

| Check | Result |
|---|---|
| `Data/Raw` unchanged since before processing (git) | pass |
| Model-table inputs (day-ahead, imbalance long and short, load forecast, wind and solar forecasts, scheduled imports and exports) | 0 invented, 0 lost, max difference at most 0.05 (CSV rounding); the 34 wind-onshore and 30 solar carried-forward cells equal the last real value |
| Slots within 6 hours of each file's resolution switch (checked separately by raw coverage) | all match |
| Targets, thresholds, calendar columns, split rule, scaler, design matrix | recomputed and match |
| Ex-post table: actual load, imbalance volume (equals the raw sum in all 68,404 slots), all 16 generation columns, physical flows, 8 balancing series | 0 invented values; max difference at most 0.05 |
| Alignment between series (correlation peaks at lag 0 for day-ahead vs imbalance prices, load forecast vs actual, wind and solar forecast vs actual, scheduled vs physical flows, imbalance price vs volume); solar peaks at 13h Paris time; load forecast error 2.0% of mean load | pass |
| Resolution only ever gets finer in the 18 files that feed the model table | pass |
| `missingness_by_year.csv`, `frequency_by_file_year.csv` recomputed | match |

**Known limits (all in the ex-post table, none affect the predictors or targets):**
1. `physical_flows_*_IT_NORD` is 15-minute on 30 Dec 2024 and hourly again on 31 Dec. The pipeline leaves 31 Dec blank for it, so `physical_imports`, `physical_exports` and `net_physical_imports` are blank for all 46 slots of that day.
2. The balancing-price columns are blank where only one of the slot's two quarter-hours has a price: 5 to 23 slots for FCR, RR and aFRR, about 150 for each mFRR direction.
3. Two imbalance-price slots (2021-01-10 11:30 and 2023-11-24 13:30) sit next to missing rows; the raw file has no row for them and the table correctly leaves them blank.
4. The raw files carry no publication times, so "known before delivery" for the day-ahead forecasts and schedules rests on their labels.

## 14. Change to the ex-post table (2026-11-21, at Vivek's request)

`expost_table.csv` now has 68,325 rows instead of 70,128. Slots were removed where the load forecast error is missing (70 slots) or `S_long` is missing (1,734 slots; 1 slot is in both lists), 1,803 in total. A new column `load_forecast` (the day-ahead load forecast, MW, placed right after `actual_load`) was added, so `load_forecast_error = actual_load - load_forecast` can be checked directly. Only the ex-post table changed. `modelling_table.csv` and `design_matrix.csv` are byte-identical to before, and the modelling table keeps its full regular 30-minute grid. Because rows were removed, the ex-post table no longer has a regular 30-minute step. Kept rows by split: train 34,401, validation 16,443, test 17,481.

## 15. Second change to the ex-post table (2026-11-21, at Vivek's request)

Slots with a missing wind onshore forecast error (260) or a missing solar forecast error (246) were also removed from `expost_table.csv`, on top of the section 14 removals. In total 2,157 of 70,128 slots are now removed, leaving 67,971 rows (train 34,097, validation 16,393, test 17,481). The columns `wind_onshore_forecast` and `solar_forecast` (day-ahead forecasts, MW) were added right after `gen_wind_onshore` and `gen_solar`, so each actual, forecast and error sits side by side; the load forecast columns from section 14 work the same way. The table now has 41 columns. `modelling_table.csv` and `design_matrix.csv` are byte-identical to before.

## 16. Thresholds and metrics recomputed on the clean data (2026-11-21, at Vivek's request)

The clean data is the set of slots kept in the ex-post table: slots that have `S_long` and the load, wind onshore and solar forecast errors (67,971 of 70,128; 2,157 removed). Everything below is now computed on those slots only, using the 2021-22 training part for the thresholds and the scaler.

| Quantity | Before | Now |
|---|---|---|
| `q90_short` | 77.890 | **77.824** |
| `q90_long` | 78.011 | **78.218** |

| Split | Slots | Clean-data slots | Rows in design matrix | Share `Y_short = 1` | Share `Y_long = 1` |
|---|---|---|---|---|---|
| train (2021-22) | 35,040 | 34,097 | 34,053 | 10.0% | 10.0% |
| validation (2023) | 17,520 | 16,393 | 16,393 | 7.2% | 5.3% |
| test (2024) | 17,568 | 17,481 | 17,481 | 10.4% | 8.1% |

The shares are the share of `Y = 1` among the clean-data slots of each split. The design matrix now has 67,927 rows (clean slots with complete predictors and both targets), and `scaler.json` holds the training means and standard deviations of those rows (for example day-ahead price mean 194.43, SD 145.38). *(Superseded by section 17: `modelling_table.csv` is now cut to the clean slots and has no `in_clean_data` column.)* Labels `Y_short` and `Y_long` are assigned to every slot that has a stress value using the new thresholds, so they are identical in all three tables.

Checked: the thresholds equal the 90th percentile of the clean training slots (independent recompute), every label equals `S > q90` on all 68,394 slots that have a stress value, the training positive rate is exactly 10%, the scaler matches the clean complete training rows, and rebuilding gives identical files.

## 17. Modelling table cut to the clean slots (2026-11-21, at Vivek's request)

`modelling_table.csv` now holds exactly the same 67,971 slots as `expost_table.csv` (train 34,097, validation 16,393, test 17,481) and 23 columns; the `in_clean_data` column was dropped because every remaining row is clean. The two tables line up row for row, and their 9 shared columns are identical on every row. The pipeline now fails if any of `day_ahead_price`, `price_long`, `price_short`, `load_forecast`, `wind_onshore_forecast`, `solar_forecast`, `D_*`, `S_*` or `Y_*` has a gap in this table, and none does. Two columns still have gaps outside what was removed: `scheduled_imports` and `net_scheduled_imports` are missing in 44 slots (all in training), which is why `design_matrix.csv` has 67,927 rows. Thresholds, labels, scaler, design matrix and the ex-post table are unchanged by this step. Because rows were deleted, the modelling table no longer has a regular 30-minute step; the full-grid audit results in sections 10 and 13 describe the grid before this cut.

## 18. Offshore wind no longer filled with zero (2026-11-21, at Vivek's request)

The earlier default (section 11, item 2) filled `wind_offshore_forecast` with 0 before its first report on 2023-08-06. That is removed: the column is now blank wherever the raw file is blank (all of 2021-22 and 58% of 2023; 43,669 of the 67,971 modelling-table slots), and only a genuine reported 0 stays 0 (856 slots). A code search found no other place that fills blanks with zero. The other fills are not zeros: forecasts are carried forward at most 4 slots (2 hours), and border totals are blank if any of the 6 borders is blank.

Knock-on effects: `residual_load_forecast` subtracts offshore wind, so it is now blank in the same 43,669 slots and blank for the whole training period; the scaler records it as all-missing in training and the design matrix no longer contains it. `wind_offshore_forecast` is still not in the design matrix (blank in training). It is also no longer a required predictor when counting usable rows, so the design matrix keeps 67,927 rows. Thresholds, labels and the ex-post table are unchanged.
