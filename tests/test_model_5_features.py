"""Model 5's rolling features may only use information from before the delivery day.

Run from the repository root: `python -m pytest tests/test_model_5_features.py -q`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import model_5_rolling as m5  # noqa: E402


@pytest.fixture(scope="module")
def data():
    table = pd.read_csv(ROOT / "Data" / "Processed" / "modelling_table.csv", index_col=0)
    matrix = pd.read_csv(ROOT / "Data" / "Processed" / "design_matrix.csv", index_col=0)
    local = pd.to_datetime(table["datetime_paris"], utc=True).dt.tz_convert("Europe/Paris")
    day = pd.to_datetime(local.dt.strftime("%Y-%m-%d")).to_numpy()
    return table, matrix, local, day, m5.build_features(table, matrix)[1]


def test_features_do_not_change_when_the_future_is_removed(data):
    table, matrix, local, _, features = data
    keep = (local < pd.Timestamp("2023-07-01", tz="Europe/Paris")).to_numpy()
    cut = m5.build_features(table[keep], matrix[keep])[1]
    full = features[keep]
    changed = ((cut - full).abs() > 1e-9) | (cut.isna() != full.isna())
    assert not changed.any(axis=None)


def test_stress_history_is_lagged_by_two_days(data):
    table, matrix, _, day, features = data
    d0 = pd.Timestamp("2024-03-15")
    altered = table.copy()
    flip = np.isin(day, [d0 - pd.Timedelta(days=1), d0])
    for col in ("Y_short", "Y_long"):
        altered.loc[flip, col] = 1 - altered.loc[flip, col]
    new = m5.build_features(altered, matrix)[1]
    same_day = day == d0
    assert np.allclose(new.loc[same_day], features.loc[same_day], equal_nan=True)  # D and D-1 unseen on D
    next_day = day == d0 + pd.Timedelta(days=1)
    assert not np.allclose(new.loc[next_day, "ydr_short"], features.loc[next_day, "ydr_short"])  # D-1 seen on D+1


def test_trailing_price_average_excludes_the_day_itself(data):
    table, matrix, _, day, features = data
    d0 = pd.Timestamp("2024-03-15")
    altered = table.copy()
    altered.loc[day == d0, "day_ahead_price"] += 500
    new = m5.build_features(altered, matrix)[1]
    rows = day == d0
    before = (table["day_ahead_price"] - features["price_dev7"]).loc[rows]
    after = (altered["day_ahead_price"] - new["price_dev7"]).loc[rows]
    assert np.allclose(before, after, equal_nan=True)


def test_test_rows_are_the_same_for_every_model(data):
    _, matrix, local, _, _ = data
    assert ((local.dt.year == 2024).to_numpy() == (matrix["split"] == "test").to_numpy()).all()
