"""Model 4 adds Model 2's HMM regime probabilities to the Model 3 XGBoost inputs.

Run from the repository root with `python -m src.models.model_4_xgboost_hmm`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ..data_downloads_and_manipulation.build_dataset import RAW, build_day_ahead_table
from .model_1_logistic_regression import FEATURES, ROOT
from .model_2_logistic_hmm import add_regime_features
from .model_3_xgboost import DESIGN_MATRIX, run_xgboost_models

OUT = ROOT / "results" / "model_4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-matrix", type=Path, default=DESIGN_MATRIX,
                        help="Path to the shared 2021–2024 design_matrix.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW,
                        help="Directory containing the 2021–2024 day-ahead ENTSO-E CSVs")
    args = parser.parse_args()
    matrix = pd.read_csv(args.design_matrix, index_col=0)
    _, day_ahead = build_day_ahead_table(args.raw_dir)
    # Model 2 supplies the features, and Model 3 supplies the identical search and scoring protocol.
    matrix, regime_features, summary = add_regime_features(matrix, day_ahead)
    run_xgboost_models(matrix, FEATURES + regime_features, OUT)
    (OUT / "regimes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
