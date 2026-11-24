# French electricity imbalance-settlement stress

Using only information available after the French day-ahead market has cleared and before delivery, can market fundamentals identify the 30-minute intervals of unusually severe imbalance-settlement stress, separately for short and for long positions? Five models are compared (logistic regression, logistic regression with hidden-Markov regimes, gradient-boosted trees, boosted trees with regimes, and a model with rolling features). The data are ENTSO-E Transparency Platform series for France, 1 January 2021 to 31 December 2024.

The full write-up is the notebook `report/report.ipynb`.

## Repository layout

| Path | Content |
|---|---|
| `report/report.ipynb` | The report: question, data, descriptive analysis, models, results, conclusion |
| `src/data_downloads_and_manipulation/` | Download scripts, `build_dataset.py` (cleaning and labels), `check_api_key.py` |
| `src/models/` | Models 1 to 5 |
| `src/diagrams.py` | The model diagrams drawn in the notebook |
| `tests/` | Leakage tests and tests of the Model 5 features |
| `Data/Raw/`, `Data/Processed/` | Downloaded series and the tables built from them |
| `results/` | Metrics, settings and coefficients saved by each model |
| `notes/` | Working notes and change logs |

## Setup

Python 3.11 is recommended.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs the OpenMP library: `brew install libomp`.

## ENTSO-E API key

The download scripts need a personal API key for the ENTSO-E Transparency Platform. Register an account at https://transparency.entsoe.eu and ask for API access by email to transparency@entsoe.eu. Then create a file called `.env` in the repository root with one line:

```
ENTSOE_API_KEY=your-key-here
```

`.env` is listed in `.gitignore`, so the key is never committed. To check that the key works:

```
python src/data_downloads_and_manipulation/check_api_key.py
```

## Running the project

Run everything from the repository root, in this order.

1. Download the raw data (long, needs the API key):
   ```
   python src/data_downloads_and_manipulation/download_entsoe.py
   python src/data_downloads_and_manipulation/download_balancing.py
   ```
2. Build the modelling tables (writes `Data/Processed/`):
   ```
   python src/data_downloads_and_manipulation/build_dataset.py
   ```
3. Fit the models, in order (each writes `results/model_k/`). Model 5 reads the saved settings of Models 1 to 4, so run it last:
   ```
   python -m src.models.model_1_logistic_regression
   python -m src.models.model_2_logistic_hmm
   python -m src.models.model_3_xgboost
   python -m src.models.model_4_xgboost_hmm
   python -m src.models.model_5_rolling
   ```
4. Open `report/report.ipynb` and run all cells. The notebook reads the saved tables, so it runs in seconds. Section 4.5 lists the scripts above and can run steps 2 and 3 from the notebook (set `RUN_BUILD_AND_MODELS = True`).

To run the tests:

```
python -m pytest tests -q
```

## A note on the data in Git

`Data/` is about 144 MB (105 MB raw, 39 MB processed) and is tracked in Git, so the notebook and the models run straight after cloning, without an API key. The download scripts can rebuild all of it. The raw files are never modified by the build step.
