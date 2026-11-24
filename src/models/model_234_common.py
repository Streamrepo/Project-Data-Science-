"""Shared feature contract and scoring for Models 2–4 on the original design matrix."""

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DESIGN_MATRIX = ROOT / "Data" / "Processed" / "design_matrix.csv"
RAW = ROOT / "Data" / "Raw" / "ENSTOE"
TARGETS = ("Y_short", "Y_long")
SPLITS = ("train", "validation", "test")

# Match the 17 Model 1 predictors. The matrix already contains training-scaled fundamentals.
FEATURES = [
    "day_ahead_price_z", "load_forecast_z", "wind_onshore_forecast_z", "solar_forecast_z",
    "scheduled_imports_z", "scheduled_exports_z", "hour_sin", "hour_cos",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "season_spring", "season_summer", "season_autumn",
]


def evaluate(labels, probabilities, train_rate):
    """Score ranking, probability accuracy and precision at a fixed screening budget."""
    # Use the training event rate for the constant forecast. AP summarizes rare-event ranking.
    constant = np.full_like(probabilities, train_rate)
    # Keep chronological order among tied scores so top-decile comparisons are reproducible.
    top_count = max(1, int(round(0.10 * len(probabilities))))
    top = np.argsort(-probabilities, kind="stable")[:top_count]
    return {
        "rows": len(labels),
        "base_rate": float(labels.mean()),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "brier_constant": brier_score_loss(labels, constant),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "log_loss_constant": log_loss(labels, constant, labels=[0, 1]),
        "precision_top10": float(labels[top].mean()),
    }
