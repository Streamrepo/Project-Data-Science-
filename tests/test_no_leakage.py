"""Leakage audit: automated checks that no model input uses information from after the cutoff.

Run from the repository root: `python -m pytest tests/test_no_leakage.py -q`
"""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "Data" / "Processed"
RAW = ROOT / "Data" / "Raw" / "ENSTOE"

spec = importlib.util.spec_from_file_location(
    "build_dataset", ROOT / "src" / "data_downloads_and_manipulation" / "build_dataset.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.models import model_1_logistic_regression as m1  # noqa: E402
from src.models import model_2_logistic_hmm as m2  # noqa: E402
from src.models.model_234_common import FEATURES as FEATURES_234  # noqa: E402

DAY_AHEAD = {"day_ahead_price_z", "load_forecast_z", "wind_onshore_forecast_z", "solar_forecast_z",
             "scheduled_imports_z", "scheduled_exports_z"}
CALENDAR = {"hour_sin", "hour_cos", *(f"dow_{d}" for d in range(1, 7)),
            "season_spring", "season_summer", "season_autumn"}
POST_DELIVERY_PREFIXES = ("S_", "D_", "Y_", "price_", "imbalance", "bal_", "gen_", "actual_",
                          "physical_")


@pytest.fixture(scope="module")
def matrix():
    return pd.read_csv(PROC / "design_matrix.csv", index_col=0)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(PROC / "modelling_table.csv", index_col=0)


def test_features_are_a_day_ahead_whitelist():
    for features in (m1.FEATURES, FEATURES_234):
        assert set(features) <= DAY_AHEAD | CALENDAR
        assert not any(f.startswith(POST_DELIVERY_PREFIXES) for f in features)
    assert list(m1.FEATURES) == list(FEATURES_234)


def test_design_matrix_features_exclude_post_delivery_columns(matrix):
    inputs = [c for c in matrix.columns if c not in ("datetime_paris", "split", "Y_short", "Y_long")]
    derived = {"net_scheduled_imports_z", "residual_load_forecast_z", "wind_offshore_forecast_z"}  # day-ahead, unused by models
    assert set(inputs) <= DAY_AHEAD | CALENDAR | derived


def test_splits_are_chronological_and_disjoint(matrix):
    t = pd.to_datetime(matrix["datetime_paris"], utc=True)
    bounds = {s: (t[matrix.split == s].min(), t[matrix.split == s].max())
              for s in ("train", "validation", "test")}
    assert bounds["train"][1] < bounds["validation"][0]
    assert bounds["validation"][1] < bounds["test"][0]
    year = t.dt.tz_convert("Europe/Paris").dt.year
    assert set(year[matrix.split == "train"]) == {2021, 2022}
    assert set(year[matrix.split == "validation"]) == {2023}
    assert set(year[matrix.split == "test"]) == {2024}


def test_scaler_uses_training_rows_only(table):
    scaler = json.loads((PROC / "scaler.json").read_text())
    train = table[table["split"] == "train"]
    for col in ("day_ahead_price", "load_forecast", "solar_forecast"):
        assert scaler[col]["mean"] == pytest.approx(train[col].mean(), rel=1e-6)
        assert scaler[col]["sd"] == pytest.approx(train[col].std(ddof=0), rel=1e-6)
        assert scaler[col]["mean"] != pytest.approx(table[col].mean(), rel=1e-4)  # not all years


def test_thresholds_use_training_rows_only(table):
    thr = json.loads((PROC / "thresholds.json").read_text())
    train = table[table["split"] == "train"]
    assert thr["q90_short"] == pytest.approx(train["S_short"].quantile(0.9))
    assert thr["q90_long"] == pytest.approx(train["S_long"].quantile(0.9))
    assert thr["q90_short"] != pytest.approx(table["S_short"].quantile(0.9), abs=1e-6)


def test_cohort_is_defined_without_post_delivery_actuals(table):
    expost = pd.read_csv(PROC / "expost_table.csv", index_col=0)
    # rows exist in the model although their post-delivery actuals are missing
    missing_actuals = expost["load_forecast_error"].isna() | expost["solar_forecast_error"].isna() \
        | expost["wind_onshore_forecast_error"].isna()
    assert missing_actuals.sum() > 0
    assert set(expost.index[missing_actuals]) <= set(table.index)
    # and the cohort is exactly: day-ahead inputs present and both targets present
    assert table[bd.X_REQUIRED + ["S_short", "S_long"]].notna().all().all()


def test_cadence_inference_is_causal():
    """The cadence at row i must not change if every later row is removed."""
    frame = pd.read_csv(RAW / "day_ahead_prices_fr.csv", index_col=0)
    index = pd.to_datetime(frame.index, utc=True).sort_values().unique()
    index = pd.DatetimeIndex(index)
    full = bd.covered_cells(index)
    for cut in (5_000, 20_000, 40_000, len(index) - 1):
        assert (bd.covered_cells(index[:cut]) == full[:cut])[48:].all(), cut


def test_regime_filter_is_causal():
    """Filtered state probabilities at t must ignore every observation after t."""
    rng = np.random.default_rng(0)
    hmm = types.SimpleNamespace(
        covars_=np.stack([np.eye(3) * v for v in (0.5, 1.0, 2.0)]),
        means_=rng.normal(size=(3, 3)), startprob_=np.full(3, 1 / 3),
        transmat_=np.array([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]]), n_components=3)
    obs = rng.normal(size=(300, 3))
    full = m2.filter_regimes(hmm, obs)
    changed = obs.copy()
    changed[200:] += 10  # alter the future
    np.testing.assert_allclose(m2.filter_regimes(hmm, changed)[:200], full[:200])


def test_future_rows_do_not_change_earlier_scaling(table):
    """Standardising with training statistics: appending future data cannot move earlier z-scores."""
    scaler = json.loads((PROC / "scaler.json").read_text())
    matrix = pd.read_csv(PROC / "design_matrix.csv", index_col=0)
    idx = matrix.index[matrix.split == "validation"][:100]
    z = (table.loc[idx, "day_ahead_price"] - scaler["day_ahead_price"]["mean"]) \
        / scaler["day_ahead_price"]["sd"]
    np.testing.assert_allclose(z.to_numpy(), matrix.loc[idx, "day_ahead_price_z"].to_numpy(), atol=1e-4)
