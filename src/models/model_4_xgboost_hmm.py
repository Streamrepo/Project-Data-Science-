"""Model 4 adds Model 2's HMM regime probabilities to the Model 3 XGBoost inputs.

Run from the repository root with `python -m src.models.model_4_xgboost_hmm`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .model_234_common import DESIGN_MATRIX, FEATURES, RAW, ROOT
from .model_2_logistic_hmm import add_regime_features, load_day_ahead
from .model_3_xgboost import run_xgboost_models

OUT = ROOT / "results" / "model_4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-matrix", type=Path, default=DESIGN_MATRIX,
                        help="Path to the shared 2021–2024 design_matrix.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW,
                        help="Directory containing the 2021–2024 day-ahead ENTSO-E CSVs")
    args = parser.parse_args()
    matrix = pd.read_csv(args.design_matrix, index_col=0)
    day_ahead = load_day_ahead(args.raw_dir)
    # Change only the inputs relative to Model 3 so the comparison isolates the addition of regimes.
    matrix, regime_features, summary = add_regime_features(matrix, day_ahead)
    run_xgboost_models(matrix, FEATURES + regime_features, OUT)
    (OUT / "regimes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
