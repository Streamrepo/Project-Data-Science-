"""Model 1 (Modelling_Plan.md section 9): two separate logistic regressions that predict whether
the stress S is above its training-only 90th percentile:

    Y_short = 1 if S_short > q90_short   (S_short = P_short - P_DA)
    Y_long  = 1 if S_long  > q90_long    (S_long  = P_DA - P_long)

using only information known after the day-ahead auction (no lags, no same-period realised values).

Input : Data/Processed/design_matrix.csv  (scaled with training-only means and SDs, has Y_short/Y_long)
Output: results/model_1/  metrics.csv, coefficients.csv, c_search.csv

Design (plan section 11 defaults):
- L2 penalty; its strength C is chosen on the 2023 validation split only (best PR-AUC).
- Unweighted classes.
- Fitted on 2021-22 training rows only; 2024 test is reported, never used for choices.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "Data" / "Processed"
OUT = ROOT / "results" / "model_1"

# Explanatory variables: all known after the day-ahead auction, before delivery.
# net_scheduled_imports_z is left out: it is exactly imports - exports (perfect collinearity).
# wind_offshore_forecast and residual_load_forecast are blank for the whole training period.
FEATURES = [
    "day_ahead_price_z", "load_forecast_z", "wind_onshore_forecast_z", "solar_forecast_z",
    "scheduled_imports_z", "scheduled_exports_z",
    "hour_sin", "hour_cos",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",          # Monday = reference
    "season_spring", "season_summer", "season_autumn",             # winter = reference
]
TARGETS = ["Y_short", "Y_long"]
C_GRID = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
SPLITS = ["train", "validation", "test"]


def precision_top(y, p, share=0.10):
    """Share of Y = 1 among the highest-scoring `share` of intervals."""
    k = max(1, int(round(share * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean())


def evaluate(y, p, train_rate):
    base = np.full_like(p, train_rate)
    return {
        "rows": len(y),
        "base_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "brier_constant": brier_score_loss(y, base),   # always predicting the training rate
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "log_loss_constant": log_loss(y, base, labels=[0, 1]),
        "precision_top10": precision_top(y, p),
    }


def fit(X, y, C):
    # L2 is scikit-learn's default penalty (l1_ratio=0); C is the inverse penalty strength.
    return LogisticRegression(C=C, solver="lbfgs", max_iter=5000).fit(X, y)


def main():
    df = pd.read_csv(PROCESSED / "design_matrix.csv", index_col=0)
    q90 = json.loads((PROCESSED / "thresholds.json").read_text())
    assert df[FEATURES + TARGETS].notna().all().all(), "gaps in the design matrix"
    assert not any(c.startswith(("S_", "D_", "Y_", "price")) for c in FEATURES), "leakage"
    part = {s: df[df["split"] == s] for s in SPLITS}
    OUT.mkdir(parents=True, exist_ok=True)

    metrics, coefs, search = [], [], []
    for target in TARGETS:
        Xtr, ytr = part["train"][FEATURES].to_numpy(), part["train"][target].to_numpy(int)
        Xva, yva = part["validation"][FEATURES].to_numpy(), part["validation"][target].to_numpy(int)

        # Choose C on validation only.
        for C in C_GRID:
            p = fit(Xtr, ytr, C).predict_proba(Xva)[:, 1]
            search.append({"target": target, "C": C, "val_pr_auc": average_precision_score(yva, p),
                           "val_log_loss": log_loss(yva, p, labels=[0, 1])})
        cand = [r for r in search if r["target"] == target]
        best_C = max(cand, key=lambda r: r["val_pr_auc"])["C"]

        model = fit(Xtr, ytr, best_C)
        train_rate = float(ytr.mean())
        for s in SPLITS:
            y = part[s][target].to_numpy(int)
            p = model.predict_proba(part[s][FEATURES].to_numpy())[:, 1]
            metrics.append({"target": target, "split": s, "C": best_C, **evaluate(y, p, train_rate)})
        coefs.append(pd.Series(np.r_[model.intercept_, model.coef_[0]],
                               index=["intercept"] + FEATURES, name=target))

    metrics = pd.DataFrame(metrics)
    coefs = pd.concat(coefs, axis=1)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    coefs.to_csv(OUT / "coefficients.csv")
    pd.DataFrame(search).to_csv(OUT / "c_search.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Thresholds (training-only q90 of S, EUR/MWh): {q90}")
    print("Metrics (constant = always predicting the training positive rate):")
    print(metrics.round(4).to_string(index=False))
    print("\nCoefficients (log-odds per 1 SD for *_z, per unit otherwise):")
    print(coefs.round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
