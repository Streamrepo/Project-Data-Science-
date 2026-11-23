# Luc's changes

Changes on the `Lucs-Magic-Tree` branch that Luc has confirmed. Each entry is a short summary of what was done; newest at the bottom.

| Date | Change |
|---|---|
| 2026-11-23 | Moved `download_balancing.py`, `download_entsoe.py` and `test.py` into `src/`; their paths now point to the project root. |
| 2026-11-23 | Deleted the empty `Description_ofwhatisdone.md`. |
| 2026-11-23 | Moved all four scripts into `src/data_downloads_and_manipulation/` (root paths updated, rebuild identical) and added an empty `src/models/` folder. |
| 2026-11-23 | Updated the script paths in `Modelling_Plan.md` to `src/data_downloads_and_manipulation/`. |
| 2026-11-23 | Added `report/report.ipynb`, a starter Jupyter notebook for the project report. |
| 2026-11-23 | Removed the 44 rows (3 Feb 2021) missing `scheduled_imports` and `net_scheduled_imports` from the modelling and ex-post tables, leaving no gaps (67,927 rows; q90 now 77.86 short, 78.27 long). |
| 2026-11-23 | Added Model 1 (`src/models/model_1_logistic_regression.py`): two L2 logistic regressions for S_short/S_long above q90 on 17 day-ahead inputs; test ROC-AUC 0.53 short, 0.58 long. |
| 2026-11-23 | Filled `report/report.ipynb`: economic reasoning for S, distributions, Model 1 inputs/response, recomputed q90 check, Y = 1 counts, S_short/S_long correlation (-0.99) and a model diagram. |
| 2026-11-23 | Added Model 1 and report sections (19-21) to `Modelling_Plan.md` and updated its status line. |
