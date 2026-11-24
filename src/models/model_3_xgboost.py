"""Model 3 predicts each severe-stress event with boosted trees and day-ahead fundamentals.

Run from the repository root with `python -m src.models.model_3_xgboost`.
"""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from xgboost import XGBClassifier

from .model_234_common import DESIGN_MATRIX, FEATURES, ROOT, SPLITS, TARGETS, evaluate

OUT = ROOT / "results" / "model_3"
DEPTHS = (2, 3, 5)
LEARNING_RATES = (0.03, 0.1)
TREE_COUNTS = (100, 300)


def run_xgboost_models(matrix, features, out):
    """Train on 2021–2022 and choose the settings for each target on 2023 alone."""
    if matrix[[*features, *TARGETS]].isna().any().any():
        raise ValueError("Missing predictors or targets in the design matrix")

    # The processed matrix fixes the original rows, thresholds and chronological split.
    parts = {split: matrix.loc[matrix["split"] == split] for split in SPLITS}
    predictors = {split: part[features] for split, part in parts.items()}
    metrics, search, importance = [], [], {}
    for target in TARGETS:
        labels = {split: part[target].to_numpy(int) for split, part in parts.items()}
        best_score = -float("inf")
        # Depth controls interactions, while shrinkage and tree count control the ensemble fit.
        for depth, rate, trees in product(DEPTHS, LEARNING_RATES, TREE_COUNTS):
            params = {"max_depth": depth, "learning_rate": rate, "n_estimators": trees}
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                tree_method="hist",
                importance_type="gain",
                random_state=42,
                n_jobs=1,
            ).fit(predictors["train"], labels["train"])
            probabilities = model.predict_proba(predictors["validation"])[:, 1].astype(float)
            # Average precision rewards ranking rare severe intervals ahead of ordinary ones.
            score = average_precision_score(labels["validation"], probabilities)
            search.append({
                "target": target,
                **params,
                "val_pr_auc": score,
                "val_log_loss": log_loss(labels["validation"], probabilities, labels=[0, 1]),
            })
            if score > best_score:
                best_score, best_model, best_params = score, model, params
                validation_probabilities = probabilities

        # Score the selected training fit, including the held-out 2024 test and the constant benchmark.
        train_rate = labels["train"].mean()
        for split in SPLITS:
            probabilities = (validation_probabilities if split == "validation"
                             else best_model.predict_proba(predictors[split])[:, 1].astype(float))
            metrics.append({
                "target": target,
                "split": split,
                **best_params,
                **evaluate(labels[split], probabilities, train_rate),
            })
        importance[target] = best_model.feature_importances_

    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(search).to_csv(out / "parameter_search.csv", index=False)
    pd.DataFrame(importance, index=pd.Index(features, name="feature")).to_csv(
        out / "feature_importance.csv"
    )
    print(metrics.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-matrix", type=Path, default=DESIGN_MATRIX,
        help="Path to the shared design_matrix.csv (defaults to Data/Processed/design_matrix.csv)",
    )
    args = parser.parse_args()
    matrix = pd.read_csv(args.design_matrix, index_col=0)
    run_xgboost_models(matrix, FEATURES, OUT)


if __name__ == "__main__":
    main()
