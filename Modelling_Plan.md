# Modelling plan: leakage-safe baseline for French imbalance-settlement stress

**Source:** `French_Electricity_Project_Modelling_Brief (1).pdf` (8 pages). Sections 1 to 9 follow the brief. Section 10 is my addition from profiling the raw data, and section 11 lists decisions the brief leaves open. Both are labelled as not coming from the brief.

**Status (2026-11-21): plan only. Nothing from this plan is to be executed until Vivek says to start.**

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
