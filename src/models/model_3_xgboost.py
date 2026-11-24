"""Model 3 fits two XGBoost classifiers on the Model 1 fundamentals.

Run from the repository root with `python -m src.models.model_3_xgboost`.
"""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from xgboost import XGBClassifier

from .model_1_logistic_regression import FEATURES, PROCESSED, ROOT, SPLITS, TARGETS, evaluate

OUT = ROOT / "results" / "model_3"
DESIGN_MATRIX = PROCESSED / "design_matrix.csv"
PARAMETER_GRID = tuple(product((2, 3, 5), (0.03, 0.1), (100, 300)))


def run_xgboost_models(matrix, features, out):
    """Select each classifier on validation average precision and score each split."""
    if matrix[features + TARGETS].isna().any().any():
        raise ValueError("Missing predictors or targets in the design matrix")

    # Reuse the baseline rows and chronological split. The input already contains training-scaled features.
    parts = {split: matrix.loc[matrix["split"] == split] for split in SPLITS}
    predictors = {split: part[features] for split, part in parts.items()}
    metrics, search, importance = [], [], {}
    for target in TARGETS:
        labels = {split: part[target].to_numpy(int) for split, part in parts.items()}
        train_rate = labels["train"].mean()
        best_score = -float("inf")
        # Boosted trees capture thresholds and interactions. Rank candidates using 2023 AP only.
        for depth, rate, trees in PARAMETER_GRID:
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

        # Evaluate the selected model on each split. Proper scores also compare against the train rate.
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
    print(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-matrix", type=Path, default=DESIGN_MATRIX,
        help="Path to the shared design_matrix.csv (defaults to Data/Processed/design_matrix.csv)",
    )
    args = parser.parse_args()
    if not args.design_matrix.is_file():
        parser.error(f"Design matrix not found at {args.design_matrix}. Pass --design-matrix PATH.")
    matrix = pd.read_csv(args.design_matrix, index_col=0)
    run_xgboost_models(matrix, FEATURES, OUT)


if __name__ == "__main__":
    main()
