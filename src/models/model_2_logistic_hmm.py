"""Model 2 classifies short and long stress with logistic regression and HMM features.

Run from the repository root with `python -m src.models.model_2_logistic_hmm`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from .model_234_common import DESIGN_MATRIX, FEATURES, RAW, ROOT, SPLITS, TARGETS, evaluate

OUT = ROOT / "results" / "model_2"
MARKET_TZ = "Europe/Paris"
START = pd.Timestamp("2021-01-01", tz=MARKET_TZ).tz_convert("UTC")
END = pd.Timestamp("2025-01-01", tz=MARKET_TZ).tz_convert("UTC")
SLOTS = pd.date_range(START, END, freq="30min", inclusive="left", name="datetime")
QUARTER = pd.Timedelta("15min")
BORDERS = ["BE", "DE_LU", "CH", "IT_NORD", "ES", "GB"]
HMM_INPUTS = [
    "day_ahead_price",
    "load_forecast",
    "wind_onshore_forecast",
    "solar_forecast",
    "scheduled_imports",
    "scheduled_exports",
]
N_REGIMES = 3
HMM_SEEDS = (42, 43, 44)
C_GRID = (1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0)


def half_hourly(series):
    """Align raw day-ahead observations to the baseline's half-hour grid."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[(series.index >= START) & (series.index < END)]

    # Follow the baseline's cadence rule for hourly, half-hourly and quarter-hourly products.
    # A lone long gap does not turn a half-hourly observation into an hourly one.
    gaps = series.index.to_series().diff() / pd.Timedelta("1min")
    gaps = gaps.where(gaps.isin((15, 30, 60)))
    # Cadence comes from past gaps only (the finest of the previous 48), as in build_dataset.
    cadence = gaps.rolling(48, min_periods=1).min()
    quarters = (cadence.ffill().bfill() / 15).round().astype(int).replace(3, 4).to_numpy()

    # An observation covers consecutive quarters. Both quarters must exist for a half-hour mean.
    first = ((series.index - START) // QUARTER).to_numpy()
    offsets = np.arange(quarters.sum()) - np.repeat(np.cumsum(quarters) - quarters, quarters)
    positions = np.repeat(first, quarters) + offsets
    quarter_values = np.full(2 * len(SLOTS), np.nan)
    inside = positions < len(quarter_values)
    quarter_values[positions[inside]] = np.repeat(series.to_numpy(float), quarters)[inside]
    return pd.Series(quarter_values.reshape(-1, 2).mean(axis=1), index=SLOTS)


def load_day_ahead(raw_dir):
    """Read the six HMM predictors on a complete grid before selecting any labelled rows."""
    def read(name):
        frame = pd.read_csv(raw_dir / f"{name}.csv", index_col=0)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.sort_index()

    renewables = read("renewables_forecast_fr")
    imports = pd.concat([half_hourly(read(f"scheduled_{border}_FR").iloc[:, 0])
                         for border in BORDERS], axis=1)
    exports = pd.concat([half_hourly(read(f"scheduled_FR_{border}").iloc[:, 0])
                         for border in BORDERS], axis=1)
    day_ahead = pd.DataFrame({
        "day_ahead_price": half_hourly(read("day_ahead_prices_fr").iloc[:, 0]),
        "load_forecast": half_hourly(read("load_forecast_fr").iloc[:, 0]),
        "wind_onshore_forecast": half_hourly(renewables["Wind Onshore"]),
        "solar_forecast": half_hourly(renewables["Solar"]),
        "scheduled_imports": imports.sum(axis=1, min_count=len(BORDERS)),
        "scheduled_exports": exports.sum(axis=1, min_count=len(BORDERS)),
    })
    # Carry forward available forecasts for at most two hours, as in the original dataset.
    # Total schedules require all borders and the day-ahead price stays missing when unavailable.
    forecasts = HMM_INPUTS[1:]
    day_ahead[forecasts] = day_ahead[forecasts].ffill(limit=4)
    day_ahead["split"] = SLOTS.tz_convert(MARKET_TZ).year.map(
        {2021: "train", 2022: "train", 2023: "validation", 2024: "test"}
    )
    return day_ahead


def filter_regimes(hmm, observations):
    """Compute P(state_t | observations through t) on the half-hour grid."""
    # Evaluate the diagonal Gaussian emissions in log space to avoid underflow.
    variances = np.diagonal(hmm.covars_, axis1=1, axis2=2)
    complete = np.isfinite(observations).all(axis=1)
    log_emissions = np.zeros((len(observations), hmm.n_components))
    residuals = observations[complete, None, :] - hmm.means_
    log_emissions[complete] = -0.5 * np.sum(
        np.log(2 * np.pi * variances) + residuals**2 / variances, axis=2
    )

    with np.errstate(divide="ignore"):
        log_prior = np.log(hmm.startprob_)
        log_transition = np.log(hmm.transmat_)

    # Forward filtering uses observations through this slot. Full-sequence smoothing would leak future data.
    probabilities = np.empty_like(log_emissions)
    for t, log_emission in enumerate(log_emissions):
        # An incomplete observation contributes no emission update. Time still advances by one slot.
        log_posterior = log_prior + log_emission
        log_posterior -= np.logaddexp.reduce(log_posterior)
        probabilities[t] = np.exp(log_posterior)
        log_prior = np.logaddexp.reduce(log_posterior[:, None] + log_transition, axis=0)
    return probabilities


def fit_hmm(observations, lengths):
    """Select an HMM initialization by training likelihood alone."""
    # Compare three initializations on training likelihood because EM can converge to local optima.
    best_hmm, best_score = None, -np.inf
    for seed in HMM_SEEDS:
        hmm = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type="diag",
            n_iter=200,
            tol=0.1,  # Absolute log-likelihood change across the full training sample.
            random_state=seed,
            implementation="scaling",
            init_params="mc",
        )
        # Start with persistent states and let EM estimate all transition probabilities.
        hmm.startprob_ = np.full(N_REGIMES, 1 / N_REGIMES)
        hmm.transmat_ = np.full((N_REGIMES, N_REGIMES), 0.05 / (N_REGIMES - 1))
        np.fill_diagonal(hmm.transmat_, 0.95)
        hmm.fit(observations, lengths=lengths)
        score = hmm.score(observations, lengths=lengths)
        if score > best_score:
            best_hmm, best_score = hmm, score
    return best_hmm, best_score


def add_regime_features(matrix, day_ahead):
    """Fit on 2021–2022, filter the full grid and join regime features by UTC delivery time."""
    predictors = day_ahead[HMM_INPUTS]
    # Estimate the scaler only where all six training predictors are observed.
    train = (day_ahead["split"] == "train") & np.isfinite(predictors).all(axis=1)
    training = predictors.loc[train]
    mean, sd = training.mean(), training.std(ddof=0)
    observations = ((predictors - mean) / sd).to_numpy()

    # Each contiguous training run is its own sequence so EM does not infer transitions across gaps.
    train_slots = np.flatnonzero(train)
    breaks = np.flatnonzero(np.diff(train_slots) > 1) + 1
    lengths = np.diff(np.r_[0, breaks, len(train_slots)])
    hmm, training_likelihood = fit_hmm(observations[train], lengths)

    # Order states by fitted day-ahead price. Two probabilities determine the third, the reference.
    price_column = HMM_INPUTS.index("day_ahead_price")
    order = np.argsort(hmm.means_[:, price_column])
    regime_features = [f"regime_{state}_prob" for state in range(1, N_REGIMES)]
    # Filter before selecting the classifier rows, which depend on ex-post data availability.
    probabilities = pd.DataFrame(
        filter_regimes(hmm, observations)[:, order[1:]],
        index=day_ahead.index,
        columns=regime_features,
    )
    summary = {
        "inputs": HMM_INPUTS,
        "training_observations": len(training),
        "training_sequences": len(lengths),
        "training_log_likelihood": training_likelihood,
        "selected_seed": hmm.random_state,
        "emission_means": (hmm.means_[order] * sd.to_numpy() + mean.to_numpy()).tolist(),
        "initial_probabilities": hmm.startprob_[order].tolist(),
        "transition_matrix": hmm.transmat_[np.ix_(order, order)].tolist(),
    }
    matrix = matrix.set_axis(pd.to_datetime(matrix.index, utc=True), axis="index")
    return matrix.join(probabilities, validate="one_to_one"), regime_features, summary


def run_logistic_models(matrix, features, out):
    """Fit the baseline logistic specification with two additional regime probabilities."""
    if matrix[[*features, *TARGETS]].isna().any().any():
        raise ValueError("Missing predictors or targets in the modelling matrix.")
    splits = {split: matrix.loc[matrix["split"] == split] for split in SPLITS}
    predictors = {split: frame[features].to_numpy() for split, frame in splits.items()}
    metrics, search, coefficients = [], [], {}
    for target in TARGETS:
        labels = {split: frame[target].to_numpy(int) for split, frame in splits.items()}
        train_rate = float(labels["train"].mean())
        best_score = -np.inf
        # C is the inverse L2 penalty strength. Choose it by 2023 average precision.
        # Every candidate fits on 2021–2022 only, and the 2024 test does not select the model.
        for c in C_GRID:
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000)
            model.fit(predictors["train"], labels["train"])
            scores = model.predict_proba(predictors["validation"])[:, 1]
            average_precision = average_precision_score(labels["validation"], scores)
            search.append({
                "target": target, "C": c, "val_pr_auc": average_precision,
                "val_log_loss": log_loss(labels["validation"], scores, labels=[0, 1]),
            })
            if average_precision > best_score:
                best_model, best_score = model, average_precision

        for split in SPLITS:
            scores = best_model.predict_proba(predictors[split])[:, 1]
            metrics.append({
                "target": target, "split": split, "C": best_model.C,
                **evaluate(labels[split], scores, train_rate),
            })
        coefficients[target] = np.r_[best_model.intercept_, best_model.coef_[0]]

    # The coefficient table expresses conditional effects on log-odds.
    out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(metrics)
    results.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(search).to_csv(out / "c_search.csv", index=False)
    pd.DataFrame(coefficients, index=["intercept", *features]).to_csv(out / "coefficients.csv")
    print(results.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-matrix", type=Path, default=DESIGN_MATRIX)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    matrix = pd.read_csv(args.design_matrix, index_col=0)
    matrix, regime_features, summary = add_regime_features(matrix, load_day_ahead(args.raw_dir))
    run_logistic_models(matrix, FEATURES + regime_features, OUT)
    (OUT / "regimes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
