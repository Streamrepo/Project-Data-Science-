"""Model 2 adds filtered market-regime probabilities to the Model 1 inputs.

Run from the repository root with `python -m src.models.model_2_logistic_hmm`.
"""

import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from ..data_downloads_and_manipulation.build_dataset import build_day_ahead_table
from .model_1_logistic_regression import FEATURES, PROCESSED, ROOT, run_logistic_models

OUT = ROOT / "results" / "model_2"
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


def main():
    matrix = pd.read_csv(PROCESSED / "design_matrix.csv", index_col=0)
    _, day_ahead = build_day_ahead_table()
    matrix, regime_features, summary = add_regime_features(matrix, day_ahead)
    # Use the baseline logistic protocol so the comparison measures the addition of regime features.
    run_logistic_models(matrix, FEATURES + regime_features, OUT)
    (OUT / "regimes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
