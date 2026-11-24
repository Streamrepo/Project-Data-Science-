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
    # hmmlearn exposes diagonal covariances as full matrices via covars_.
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


def build_regime_features(day_ahead):
    """Fit on complete 2021–2022 predictor runs and filter the full pre-delivery history."""
    predictors = day_ahead[HMM_INPUTS]
    train = (day_ahead["split"] == "train").to_numpy() & np.isfinite(predictors).all(axis=1).to_numpy()
    training = predictors.loc[train]
    mean, sd = training.mean(), training.std(ddof=0)
    observations = ((predictors - mean) / sd).to_numpy()

    # Each contiguous training run is its own sequence so EM does not infer transitions across gaps.
    train_slots = np.flatnonzero(train)
    breaks = np.flatnonzero(np.diff(train_slots) > 1) + 1
    lengths = np.diff(np.r_[0, breaks, len(train_slots)])
    hmm, training_likelihood = fit_hmm(observations[train], lengths)

    # Stable labels ordered by ascending fitted day-ahead price. Regime 0 is the logistic reference.
    price_column = HMM_INPUTS.index("day_ahead_price")
    order = np.argsort(hmm.means_[:, price_column])
    probabilities = pd.DataFrame(
        filter_regimes(hmm, observations)[:, order],
        index=day_ahead.index,
        columns=[f"regime_{state}_prob" for state in range(N_REGIMES)],
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
    return probabilities, summary


def main():
    matrix = pd.read_csv(PROCESSED / "design_matrix.csv", index_col=0)
    matrix.index = pd.to_datetime(matrix.index, utc=True)
    _, day_ahead = build_day_ahead_table()
    probabilities, summary = build_regime_features(day_ahead)

    # Match Model 1's exact evaluation rows after filtering, since they depend on ex-post availability.
    regime_features = list(probabilities.columns[1:])
    matrix = matrix.join(probabilities[regime_features], validate="one_to_one")
    run_logistic_models(matrix, FEATURES + regime_features, OUT)
    (OUT / "regimes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
