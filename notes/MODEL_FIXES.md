# Model fixes: what changed in Models 1-4 and why

Date: 2026-11-24. Scope: the leakage audit applied to the data build and to Models 1-4 as pulled
into the repository (commit `394d6bc`, "Integrate Models 2-4 and comparative results"). Model 5
(`Model_5/`, not merged) was **not** touched and still needs a re-run on the new data.

The prediction task is unchanged: for each half-hour, is `S_short = P_short - P_DA` above its
training-period q90, and is `S_long = P_DA - P_long` above its q90? Features are still the 17
day-ahead and calendar variables. Splits are unchanged (train 2021-22, validation 2023, test 2024).

## 1. Problems found

| # | Problem | Where | Effect |
|---|---|---|---|
| L1 | **Row selection used the future.** A slot was kept only if its realised load, wind and solar *actuals* existed (published after delivery). That decided which rows were trained on and where q90 fell. | `build_dataset.py`, `main()`: `use = ~drop` | 74 rows wrongly excluded; thresholds slightly off |
| L2 | **Cadence inference looked one row ahead** (next timestamp, never its value) to decide hourly vs half-hourly. | `build_dataset.py`, `covered_cells`; copy in `model_2_logistic_hmm.py`, `half_hourly` | at most 2 slots per series differ |
| L3 | **Model 1's leakage guard did nothing.** `startswith(("S_","D_","Y_","price"))` matches no column name, so it could never fire. | `model_1_logistic_regression.py` | no protection |
| L4 | Notebook text carried stale numbers ("cleaned data", 67,927 rows). | `report/report.ipynb` | wording only |

Checked and found clean (no change needed): the 17 features (all day-ahead products or calendar),
training-only standardisation, training-only q90, penalty `C` and XGBoost settings chosen on 2023
only, chronological non-overlapping splits, forward fill limited to 4 slots (past values only),
HMM probabilities are *filtered* (forward pass), not smoothed.

Not provable from the data: the raw downloads carry no publication timestamps, so we cannot show
that forecasts were not revised after the forecast cutoff.

## 2. Code changes

### `src/data_downloads_and_manipulation/build_dataset.py` (shared by all models)
- `covered_cells`: removed the next-gap term. Cadence = finest of the previous 48 gaps, forward
  filled; `bfill` now only affects the first rows of a file (2021, training).
- `main()`: the modelling cohort is now `day-ahead predictors present AND S_short, S_long present`.
  The ex-post filter (`drop`) is kept only to describe the ex-post table and to print how many
  modelled slots lack actuals (74).
- `add_labels` docstring updated to the new cohort definition.

### Model 1 `src/models/model_1_logistic_regression.py`
- Replaced the ineffective `startswith` assertion with a **whitelist**: every feature must be one
  of the six day-ahead inputs or a calendar term, otherwise the run stops.
- Model itself (L2 logistic, 17 predictors, `C` grid 1e-5..100 chosen on 2023 AP) is unchanged.

### Model 2 `src/models/model_2_logistic_hmm.py` (Gabriel's logistic + HMM)
- `half_hourly`: cadence from past gaps only (same rule as `build_dataset.py`); removed
  `following_gap` and the `fillna(following_gap)` fallback.
- HMM (3 states, diagonal Gaussian, 3 seeds, filtered probabilities) is unchanged.

### Models 3 and 4 (XGBoost, XGBoost + HMM)
- **No source change.** They changed only because the shared data and Model 2's HMM inputs changed.

### New files
- `tests/test_no_leakage.py`: 9 automated tests (feature whitelist; matrix columns; chronological
  splits; scaler on training rows only; q90 on training rows only; cohort independent of
  post-delivery actuals; causal cadence; causal HMM filter; stable scaling). All pass. A mutation
  check confirmed the whitelist rejects injected `price_short`, `S_long`, `imbalance_volume_z`.
- `MODEL_FIXES.md` (this file).

### Notebook `report/report.ipynb`
- New section "Leakage audit (applies to every model)", a section on the changes to Models 1 and 2,
  and a cell that runs the tests live. "cleaned data" became "modelling cohort"; 67,927 became 68,001.

### Environment (not in git)
- Installed `xgboost` into `.venv` and `brew install libomp` (XGBoost needs OpenMP on macOS).
  The repo still has no `requirements.txt`; it should.

## 3. Effect on the data

| | Before | After |
|---|---|---|
| Modelling rows | 67,927 | 68,001 (+74: train +58, validation +16) |
| q90 short (EUR/MWh) | 77.86 | 77.83 |
| q90 long (EUR/MWh) | 78.268 | 78.35 |
| Train / validation / test rows | 34,053 / 16,393 / 17,481 | 34,111 / 16,409 / 17,481 |
| Test prevalence short / long | 10.36% / 8.07% | 10.36% / 8.05% |

## 4. Effect on results (ROC-AUC / average precision; before to after)

| Model | Target | Validation 2023 | Test 2024 |
|---|---|---|---|
| 1 logistic | short | 0.6045 / 0.1011 to 0.6041 / 0.1008 | 0.5319 / 0.1135 to 0.5302 / 0.1129 |
| 1 logistic | long | 0.5949 / 0.0766 to 0.5948 / 0.0766 | 0.5821 / 0.1022 to 0.5827 / 0.1022 |
| 2 logistic + HMM | short | 0.6071 / 0.1020 to 0.6070 / 0.1018 | 0.5369 / 0.1152 to 0.5357 / 0.1147 |
| 2 logistic + HMM | long | 0.5912 / 0.0775 to 0.5912 / 0.0775 | 0.5840 / 0.1032 to 0.5846 / 0.1032 |
| 3 XGBoost | short | 0.6002 / 0.0962 to 0.6005 / 0.0950 | 0.5552 / 0.1233 to 0.5542 / 0.1228 |
| 3 XGBoost | long | 0.6908 / 0.1035 to 0.6880 / 0.1019 | 0.6473 / 0.1429 to 0.6439 / 0.1399 |
| 4 XGBoost + HMM | short | 0.5987 / 0.0968 to 0.6045 / 0.0961 | 0.5661 / 0.1284 to 0.5598 / 0.1245 |
| 4 XGBoost + HMM | long | 0.6897 / 0.1032 to 0.6880 / 0.1019 | 0.6477 / 0.1426 to 0.6439 / 0.1399 |

Largest move: Model 4 short test ROC-AUC, -0.006. No ranking of models and no conclusion changed:
Model 3 remains the only consistent gain (long side); HMM additions remain within noise.

The selected hyper-parameters are unchanged (Models 1/2: C=100 short, C=0.001 long; Models 3/4:
depth 2, learning rate 0.03, 300 trees short / 100 trees long). Before/after result files are in
git history (`results/model_*/`).

## 5. Still open
- `results/Model_Results_Comparative_Synthesis.md` quotes the pre-fix numbers and needs regenerating.
- `Model_5/` must be re-run on the new data before it is merged.
- Gabriel's code was edited (Model 2) and Models 2-4 re-run; tell him before pushing.
- No confidence intervals for Models 1-4; events are clustered, so small AP gaps are not significant.
