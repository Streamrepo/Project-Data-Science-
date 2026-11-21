"""Build the processed datasets from the untouched raw ENTSO-E CSVs.

Follows "French Electricity Market Project - Modelling Brief" (sections 1, 2, 5, 6, 8, 11).

Outputs (Data/Processed/):
  modelling_table.csv      one row per 30-minute delivery interval, pre-delivery predictors + targets
  expost_table.csv         realised / ex-post variables for explaining events (never used in X)
  thresholds.json          training-only q90 thresholds
  missingness_by_year.csv  share of missing values per variable and year (audit)

Run:  .venv/bin/python src/build_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw" / "ENSTOE"
OUT = ROOT / "Data" / "Processed"

# ---- master grid: every 30-minute delivery interval of 2021-2024, in UTC -------------------
START = pd.Timestamp("2021-01-01", tz="Europe/Paris").tz_convert("UTC")
END = pd.Timestamp("2025-01-01", tz="Europe/Paris").tz_convert("UTC")  # exclusive
SLOTS = pd.date_range(START, END, freq="30min", inclusive="left")
QUARTER = pd.Timedelta("15min")
N_QUARTERS = 2 * len(SLOTS)

NEIGHBOURS = ["BE", "DE_LU", "CH", "IT_NORD", "ES", "GB"]
FFILL_LIMIT = 4  # forecast gaps: carry the last known forecast forward at most 4 slots (2 h)
SPLITS = {2021: "train", 2022: "train", 2023: "validation", 2024: "test"}  # Europe/Paris year


# ---- reading and resolution handling --------------------------------------------------------
def read_raw(name, **kwargs):
    df = pd.read_csv(RAW / f"{name}.csv", index_col=0, **kwargs)
    df.index = pd.to_datetime(df.index, utc=True)  # keep the timezone, normalise to UTC
    df.index.name = "datetime"
    return df.sort_index()


def covered_cells(index):
    """Number of 15-minute cells each observation covers (1, 2 or 4).

    An observation covers the time until the next observation. Where that gap is longer than
    an hour the data is missing, so the previous gap (the series' own resolution) is used and
    the gap stays empty. Resolution is inferred row by row, never assumed per file.
    """
    t = index.to_series()
    nxt = t.shift(-1) - t
    prv = t - t.shift(1)
    limit = pd.Timedelta("60min")
    step = nxt.where(nxt <= limit, prv.where(prv <= limit)).ffill().bfill()
    cells = (step / QUARTER).round().clip(1, 4).to_numpy().astype(int)
    return np.where(cells == 3, 4, cells)


def to_slots(series, kind="mean"):
    """Put a series on the 30-minute grid.

    kind="mean"   power (MW) and prices (EUR/MWh): hourly values apply to both half-hours,
                  15-minute values are averaged (equal-length quarters = time-weighted).
    kind="energy" energy (MWh): hourly values are split across the half-hours, 15-minute
                  values are summed.
    A half-hour is NaN unless both of its quarter-hours are known: nothing is invented.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[(s.index >= START) & (s.index < END)]
    n = covered_cells(s.index)
    base = ((s.index - START) // QUARTER).to_numpy()
    offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    pos = np.repeat(base, n) + offset
    vals = np.repeat(s.to_numpy(float), n)
    if kind == "energy":
        vals = vals / np.repeat(n, n)
    keep = pos < N_QUARTERS
    cells = np.full(N_QUARTERS, np.nan)
    cells[pos[keep]] = vals[keep]
    pair = cells.reshape(-1, 2)
    out = pair.sum(axis=1) if kind == "energy" else pair.mean(axis=1)
    out[np.isnan(pair).any(axis=1)] = np.nan
    return pd.Series(out, index=SLOTS)


def slots_from(name, column=None, kind="mean"):
    df = read_raw(name)
    return to_slots(df[column] if column else df.iloc[:, 0], kind)


# ---- predictor and target tables -----------------------------------------------------------
def build_modelling_table():
    raw = pd.DataFrame(index=SLOTS)
    raw["day_ahead_price"] = slots_from("day_ahead_prices_fr")
    raw["price_long"] = slots_from("imbalance_prices_fr", "Long")
    raw["price_short"] = slots_from("imbalance_prices_fr", "Short")
    raw["load_forecast"] = slots_from("load_forecast_fr")
    ren = read_raw("renewables_forecast_fr")
    raw["wind_onshore_forecast"] = to_slots(ren["Wind Onshore"])
    raw["wind_offshore_forecast"] = to_slots(ren["Wind Offshore"])
    raw["solar_forecast"] = to_slots(ren["Solar"])

    imports = pd.concat([slots_from(f"scheduled_{n}_FR") for n in NEIGHBOURS], axis=1)
    exports = pd.concat([slots_from(f"scheduled_FR_{n}") for n in NEIGHBOURS], axis=1)
    # a total is only valid if every border is known: a missing border is not a zero flow
    raw["scheduled_imports"] = imports.sum(axis=1, min_count=len(NEIGHBOURS))
    raw["scheduled_exports"] = exports.sum(axis=1, min_count=len(NEIGHBOURS))

    df = raw.copy()
    # Forecast gaps: carry the last known forecast forward (never backward, never from the future).
    for c in ["load_forecast", "wind_onshore_forecast", "solar_forecast",
              "scheduled_imports", "scheduled_exports"]:
        df[c] = df[c].ffill(limit=FFILL_LIMIT)
    # Offshore wind is blank only before its first reported value (no offshore farm was reporting
    # yet). This documented exception treats that leading blank as zero; later blanks stay blank.
    first = raw["wind_offshore_forecast"].first_valid_index()
    df.loc[df.index < first, "wind_offshore_forecast"] = 0.0
    df["wind_offshore_forecast"] = df["wind_offshore_forecast"].ffill(limit=FFILL_LIMIT)

    df["net_scheduled_imports"] = df["scheduled_imports"] - df["scheduled_exports"]
    df["residual_load_forecast"] = (df["load_forecast"] - df["wind_onshore_forecast"]
                                    - df["wind_offshore_forecast"] - df["solar_forecast"])

    paris = SLOTS.tz_convert("Europe/Paris")
    frac_hour = paris.hour + paris.minute / 60  # 30-minute resolution, so use the fractional hour
    df["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24)
    df["day_of_week"] = paris.dayofweek
    df["month"] = paris.month
    df["split"] = [SPLITS[y] for y in paris.year]

    # Signed deviations and direction-aligned stress (brief section 1). No absolute values.
    df["D_short"] = df["price_short"] - df["day_ahead_price"]
    df["D_long"] = df["price_long"] - df["day_ahead_price"]
    df["S_short"] = df["price_short"] - df["day_ahead_price"]
    df["S_long"] = df["day_ahead_price"] - df["price_long"]

    # Training-only q90 thresholds, then fixed for validation and test.
    train = df["split"] == "train"
    thresholds = {"q90_short": float(df.loc[train, "S_short"].quantile(0.9)),
                  "q90_long": float(df.loc[train, "S_long"].quantile(0.9))}
    for side in ("short", "long"):
        s = df[f"S_{side}"]
        df[f"Y_{side}"] = (s > thresholds[f"q90_{side}"]).astype("Int8").where(s.notna())

    df.insert(0, "datetime_paris", paris)
    df.index.name = "datetime"
    return raw, df, thresholds


# ---- ex-post table (explanation only) ------------------------------------------------------
def build_expost_table(modelling):
    ex = pd.DataFrame(index=SLOTS)
    ex["actual_load"] = slots_from("actual_load_fr")
    gen = pd.read_csv(RAW / "generation_actual_fr.csv", header=[0, 1], index_col=0)
    gen.index = pd.to_datetime(gen.index, utc=True)
    for tech, kind in gen.columns:
        name = "gen_" + tech.lower().replace(" ", "_").replace("-", "_") + (
            "__consumption" if kind == "Actual Consumption" else "")
        ex[name] = to_slots(gen[(tech, kind)])
    ex["imbalance_volume"] = slots_from("imbalance_volumes_fr", kind="energy")
    imp = pd.concat([slots_from(f"physical_flows_{n}_FR") for n in NEIGHBOURS], axis=1)
    exp = pd.concat([slots_from(f"physical_flows_FR_{n}") for n in NEIGHBOURS], axis=1)
    ex["physical_imports"] = imp.sum(axis=1, min_count=len(NEIGHBOURS))
    ex["physical_exports"] = exp.sum(axis=1, min_count=len(NEIGHBOURS))
    ex["net_physical_imports"] = ex["physical_imports"] - ex["physical_exports"]

    bal = pd.read_csv(RAW / "balancing_energy_prices_fr.csv")
    bal["datetime"] = pd.to_datetime(bal["datetime"], utc=True)
    bal = bal.groupby(["ReserveType", "Direction", "datetime"], as_index=False)["Price"].mean()
    for (rtype, direction), g in bal.groupby(["ReserveType", "Direction"]):
        ex[f"bal_{rtype}_{direction}".lower()] = to_slots(g.set_index("datetime")["Price"])

    # Forecast errors (brief section 7): realised minus day-ahead forecast.
    ex["load_forecast_error"] = ex["actual_load"] - modelling["load_forecast"]
    ex["wind_onshore_forecast_error"] = ex["gen_wind_onshore"] - modelling["wind_onshore_forecast"]
    ex["solar_forecast_error"] = ex["gen_solar"] - modelling["solar_forecast"]
    keep = modelling[["datetime_paris", "split", "S_short", "S_long", "Y_short", "Y_long"]]
    ex = keep.join(ex)
    ex.index.name = "datetime"
    return ex


# ---- audit ----------------------------------------------------------------------------------
def missingness_by_year(raw, final):
    year = SLOTS.tz_convert("Europe/Paris").year
    cols = [c for c in raw.columns if c in final.columns]
    rows = {}
    for label, frame in (("raw", raw), ("final", final)):
        for c in cols:
            rows[(c, label)] = frame[c].isna().groupby(year).mean().round(4)
    return pd.DataFrame(rows).T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, table, thresholds = build_modelling_table()
    expost = build_expost_table(table)

    table.to_csv(OUT / "modelling_table.csv", float_format="%.6g")
    expost.to_csv(OUT / "expost_table.csv", float_format="%.6g")
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    missingness_by_year(raw, table).to_csv(OUT / "missingness_by_year.csv")

    # ---- checks: fail loudly if the grid or the target construction is wrong ----
    assert table.index.is_unique and table.index.is_monotonic_increasing
    assert (table.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()
    assert len(table) == 1461 * 48, len(table)
    ok = table[["price_long", "price_short"]].dropna()
    assert (ok["price_long"] <= ok["price_short"] + 1e-9).all(), "Long > Short somewhere"
    paris = table["datetime_paris"]
    assert paris[table["split"] == "train"].max() < paris[table["split"] == "validation"].min()
    assert paris[table["split"] == "validation"].max() < paris[table["split"] == "test"].min()

    x_cols = ["day_ahead_price", "load_forecast", "wind_onshore_forecast",
              "wind_offshore_forecast", "solar_forecast", "scheduled_imports",
              "scheduled_exports"]
    x_ok = table[x_cols].notna().all(axis=1)
    y_ok = table[["Y_short", "Y_long"]].notna().all(axis=1)
    order = ["train", "validation", "test"]
    print(f"slots on grid: {len(table)}   thresholds: {thresholds}")
    summary = pd.DataFrame({
        "slots": table.groupby("split").size(),
        "missing_target": (~y_ok).groupby(table["split"]).sum(),
        "missing_predictor": (~x_ok).groupby(table["split"]).sum(),
        "usable": (x_ok & y_ok).groupby(table["split"]).sum(),
        "rate_Y_short": table.groupby("split")["Y_short"].apply(lambda s: s.astype(float).mean()),
        "rate_Y_long": table.groupby("split")["Y_long"].apply(lambda s: s.astype(float).mean()),
    }).loc[order]
    print(summary.round(4).to_string())
    print("\nmissing predictor slots by variable and split:")
    print(table[x_cols].isna().groupby(table["split"]).sum().loc[order].to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
