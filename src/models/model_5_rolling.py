"""Model 5 (candidate): drift-aware boosted trees and logistic regression, blended and calibrated.

Run from the repository root with `python -m src.models.model_5_rolling`.

Ideas, all using information available at the day-ahead cutoff:
  * rolling features: price relative to its trailing 7- and 30-day mean, residual load, and the
    share of severe half-hours over the previous week (lagged at least 2 days);
  * settings chosen on rolling-origin folds (train 2021 -> validate 2022, train 2021-22 -> validate
    2023), not on one year;
  * final fit on 2021-2023, scored once on 2024;
  * Platt calibration fitted on the out-of-fold predictions;
  * Models 1-4 are re-fitted here from their saved settings so every model is compared on the same
    2024 rows with day-block bootstrap intervals.
"""
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .model_234_common import DESIGN_MATRIX, FEATURES, RAW, ROOT, TARGETS, evaluate
from .model_2_logistic_hmm import add_regime_features, load_day_ahead

OUT = ROOT / "results" / "model_5"
NEW = ["price_dev7", "price_dev30", "residual_load",
       "ysr_short", "ydr_short", "ysr_long", "ydr_long"]
C_GRID = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
XGB_GRID = list(product((1, 2), (0.02, 0.05), (100, 200, 400), (10, 50)))  # depth, lr, trees, min_child
FOLDS = [((2021,), 2022), ((2021, 2022), 2023)]
BOOT = 1000


# ---- features ------------------------------------------------------------------------------
def build_features(table=None, matrix=None):
    """Rolling features. `table` and `matrix` can be passed in to test the code on truncated data."""
    if table is None:
        table = pd.read_csv(ROOT / "Data" / "Processed" / "modelling_table.csv", index_col=0)
        matrix = pd.read_csv(DESIGN_MATRIX, index_col=0)
    assert matrix.index.equals(table.index)
    local = pd.to_datetime(table["datetime_paris"], utc=True).dt.tz_convert("Europe/Paris")
    date = pd.to_datetime(local.dt.strftime("%Y-%m-%d")).to_numpy()
    slot = local.dt.strftime("%H:%M").to_numpy()
    days = pd.date_range(date.min(), date.max(), freq="D")

    def by_day(series):
        return pd.Series(series).groupby(date).mean().reindex(days)

    # Price relative to the previous 7 / 30 full days (day D-1 is complete when the D auction clears).
    price = by_day(table["day_ahead_price"].to_numpy())
    prev7 = price.shift(1).rolling(7, min_periods=3).mean()
    prev30 = price.shift(1).rolling(30, min_periods=10).mean()
    feats = pd.DataFrame(index=table.index)
    feats["price_dev7"] = table["day_ahead_price"].to_numpy() - prev7.reindex(date).to_numpy()
    feats["price_dev30"] = table["day_ahead_price"].to_numpy() - prev30.reindex(date).to_numpy()
    feats["residual_load"] = (table["load_forecast"] - table["wind_onshore_forecast"]
                              - table["solar_forecast"]).to_numpy()
    # Share of severe half-hours over days D-8 .. D-2 (day D-1 is not fully known at the auction).
    key = pd.MultiIndex.from_arrays([date, slot])
    for target in TARGETS:
        side = target.split("_")[1]
        grid = (pd.DataFrame({"d": date, "s": slot, "y": table[target].astype(float).to_numpy()})
                .pivot_table(index="d", columns="s", values="y", aggfunc="first").reindex(days))
        slot_rate = grid.shift(2).rolling(7, min_periods=3).mean().stack()
        day_rate = grid.mean(axis=1).shift(2).rolling(7, min_periods=3).mean()
        feats[f"ysr_{side}"] = slot_rate.reindex(key).to_numpy()
        feats[f"ydr_{side}"] = day_rate.reindex(date).to_numpy()
    year = local.dt.year.to_numpy()
    return matrix, feats, year, date


def logit_matrix(matrix, feats, fit_mask):
    """Base features plus the new ones, standardised and mean-imputed with fit-window statistics."""
    new = feats[NEW]
    mean, sd = new[fit_mask].mean(), new[fit_mask].std(ddof=0).replace(0, 1)
    z = ((new - mean) / sd).fillna(0.0)
    return pd.concat([matrix[FEATURES], z], axis=1).to_numpy(float)


def xgb_model(depth, rate, trees, child, seed=42):
    return XGBClassifier(max_depth=depth, learning_rate=rate, n_estimators=trees, min_child_weight=child,
                         subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
                         tree_method="hist", random_state=seed, n_jobs=1)


# ---- rolling-origin selection -----------------------------------------------------------------
def select(matrix, feats, year, target):
    y = matrix[target].to_numpy(int)
    xgb_x = pd.concat([matrix[FEATURES], feats[NEW]], axis=1).to_numpy(float)
    store = {}
    for fold, (train_years, val_year) in enumerate(FOLDS):
        fit, val = np.isin(year, train_years), year == val_year
        lx = logit_matrix(matrix, feats, fit)
        for c in C_GRID:
            store[("logit", c, fold)] = LogisticRegression(C=c, max_iter=5000).fit(
                lx[fit], y[fit]).predict_proba(lx[val])[:, 1]
        for cfg in XGB_GRID:
            store[("xgb", cfg, fold)] = xgb_model(*cfg).fit(
                xgb_x[fit], y[fit]).predict_proba(xgb_x[val])[:, 1]

    def fold_ap(pred, fold):
        return average_precision_score(y[year == FOLDS[fold][1]], pred)

    def mean_ap(kind, cfg):
        return np.mean([fold_ap(store[(kind, cfg, f)], f) for f in range(len(FOLDS))])

    rows = [{"target": target, "kind": kind, "config": str(cfg), "cv_ap": mean_ap(kind, cfg)}
            for (kind, cfg, f) in store if f == 0]
    best_c = max(C_GRID, key=lambda c: mean_ap("logit", c))
    best_x = max(XGB_GRID, key=lambda g: mean_ap("xgb", g))
    parts = {"logit": [store[("logit", best_c, f)] for f in range(len(FOLDS))],
             "xgb": [store[("xgb", best_x, f)] for f in range(len(FOLDS))]}
    parts["blend"] = [0.5 * (a + b) for a, b in zip(parts["logit"], parts["xgb"])]
    scores = {k: float(np.mean([fold_ap(p, f) for f, p in enumerate(v)])) for k, v in parts.items()}
    kind = max(scores, key=scores.get)  # rule fixed in advance: highest mean fold average precision
    oof = np.concatenate(parts[kind])
    oof_y = np.concatenate([y[year == v] for _, v in FOLDS])
    chosen = {"kind": kind, "C": best_c, "xgb": list(best_x), "cv_ap": scores, "target": target}
    return chosen, oof, oof_y, pd.DataFrame(rows)


def fit_predict(chosen, matrix, feats, fit, test):
    y = matrix[chosen["target"]].to_numpy(int)
    out = []
    if chosen["kind"] in ("logit", "blend"):
        lx = logit_matrix(matrix, feats, fit)
        out.append(LogisticRegression(C=chosen["C"], max_iter=5000).fit(lx[fit], y[fit]).predict_proba(lx[test])[:, 1])
    if chosen["kind"] in ("xgb", "blend"):
        xx = pd.concat([matrix[FEATURES], feats[NEW]], axis=1).to_numpy(float)
        out.append(xgb_model(*chosen["xgb"]).fit(xx[fit], y[fit]).predict_proba(xx[test])[:, 1])
    return np.mean(out, axis=0)


def logit_of(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(oof, oof_y):
    model = LogisticRegression(C=1e6).fit(logit_of(oof)[:, None], oof_y)
    return lambda p: model.predict_proba(logit_of(p)[:, None])[:, 1]


# ---- Models 1-4 re-fitted from their saved settings ------------------------------------------------
def baselines(matrix, target):
    saved = {m: pd.read_csv(ROOT / "results" / f"model_{m}" / "metrics.csv") for m in (1, 2, 3, 4)}
    mm, regimes, _ = add_regime_features(matrix.copy(), load_day_ahead(RAW))
    mm.index = matrix.index
    y = matrix[target].to_numpy(int)
    split = matrix["split"].to_numpy()
    tr, te = split == "train", split == "test"
    preds = {}
    for m in (1, 2, 3, 4):
        row = saved[m][(saved[m]["target"] == target) & (saved[m]["split"] == "test")].iloc[0]
        X = mm[FEATURES + (regimes if m in (2, 4) else [])].to_numpy(float)
        if m in (1, 2):
            model = LogisticRegression(C=float(row["C"]), solver="lbfgs", max_iter=5000)
        else:
            model = XGBClassifier(max_depth=int(row["max_depth"]), learning_rate=float(row["learning_rate"]),
                                  n_estimators=int(row["n_estimators"]), objective="binary:logistic",
                                  tree_method="hist", importance_type="gain", random_state=42, n_jobs=1)
        p = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        assert abs(average_precision_score(y[te], p) - row["pr_auc"]) < 2e-3, (m, target)  # reproduces saved run
        preds[f"Model {m}"] = p
    return preds, saved[3]


def bootstrap(y, scores, dates, reference):
    """Resample whole days. Returns reference minus each other model, for AP and ROC-AUC."""
    rng = np.random.default_rng(0)
    unique = np.unique(dates)
    index = {d: np.flatnonzero(dates == d) for d in unique}
    diffs = {(n, k): [] for n in scores if n != reference for k in ("ap", "roc")}
    for _ in range(BOOT):
        pick = np.concatenate([index[d] for d in rng.choice(unique, len(unique))])
        yy = y[pick]
        if yy.min() == yy.max():
            continue
        ref_ap, ref_roc = (average_precision_score(yy, scores[reference][pick]),
                           roc_auc_score(yy, scores[reference][pick]))
        for n in scores:
            if n != reference:
                diffs[(n, "ap")].append(ref_ap - average_precision_score(yy, scores[n][pick]))
                diffs[(n, "roc")].append(ref_roc - roc_auc_score(yy, scores[n][pick]))
    return diffs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, feats, year, date = build_features()
    train22, test, fit_all = np.isin(year, (2021, 2022)), year == 2024, year <= 2023
    table, boot_rows, cv_all, selected, metrics = [], [], [], {}, []
    for target in TARGETS:
        y = matrix[target].to_numpy(int)
        chosen, oof, oof_y, cv = select(matrix, feats, year, target)
        selected[target] = {k: v for k, v in chosen.items() if k != "target"}
        cv_all.append(cv)
        print(f"\n{target}: selected {chosen['kind']}  cv AP {chosen['cv_ap']}", flush=True)
        raw = fit_predict(chosen, matrix, feats, fit_all, test)
        cal = platt(oof, oof_y)(raw)
        no_refit = fit_predict(chosen, matrix, feats, train22, test)
        base_preds, m3 = baselines(matrix, target)
        # control: Model 3's own settings and plain 17 features, refitted on 2021-2023
        r3 = m3[(m3["target"] == target) & (m3["split"] == "test")].iloc[0]
        x17 = matrix[FEATURES].to_numpy(float)
        control = XGBClassifier(max_depth=int(r3["max_depth"]), learning_rate=float(r3["learning_rate"]),
                                n_estimators=int(r3["n_estimators"]), objective="binary:logistic",
                                tree_method="hist", random_state=42, n_jobs=1
                                ).fit(x17[fit_all], y[fit_all]).predict_proba(x17[test])[:, 1]
        # Same layout as Models 1-4: train = 2021-22 rows, validation = 2023, both from the model
        # fitted on 2021-22 (the second rolling fold, uncalibrated); test = final model, 2024.
        used = {"kind": chosen["kind"]}
        if chosen["kind"] in ("logit", "blend"):
            used["C"] = chosen["C"]
        if chosen["kind"] in ("xgb", "blend"):
            used.update(zip(("max_depth", "learning_rate", "n_estimators", "min_child_weight"), chosen["xgb"]))
        rate = float(y[train22].mean())  # constant forecast: the 2021-22 event rate, as for Models 1-4
        val = year == 2023
        for split, mask, p in (("train", train22, fit_predict(chosen, matrix, feats, train22, train22)),
                               ("validation", val, fit_predict(chosen, matrix, feats, train22, val)),
                               ("test", test, cal)):
            metrics.append({"target": target, "split": split, **used, **evaluate(y[mask], p, rate)})
        scores = {**base_preds, "Model 3 settings, refit 2021-23": control,
                  "Model 5 trained 2021-22 only": no_refit, "Model 5": cal}
        for name, p in {**scores, "Model 5 uncalibrated": raw}.items():
            table.append({"target": target, "model": name, **evaluate(y[test], p, rate)})
        diffs = bootstrap(y[test], scores, date[test], reference="Model 5")
        for (name, metric), vals in diffs.items():
            v = np.array(vals)
            boot_rows.append({"target": target, "model_5_minus": name, "metric": metric,
                              "mean_diff": v.mean(), "ci_low": np.percentile(v, 2.5),
                              "ci_high": np.percentile(v, 97.5), "share_above_0": float((v > 0).mean())})
    result, boot = pd.DataFrame(table), pd.DataFrame(boot_rows)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    result.to_csv(OUT / "variants_test_metrics.csv", index=False)
    pd.concat(cv_all).to_csv(OUT / "cv_results.csv", index=False)
    boot.to_csv(OUT / "bootstrap_differences.csv", index=False)
    (OUT / "selected.json").write_text(json.dumps(selected, indent=2, default=float))
    pd.set_option("display.width", 230)
    cols = ["target", "model", "roc_auc", "pr_auc", "brier", "brier_constant", "log_loss",
            "log_loss_constant", "precision_top10"]
    print("\n=== 2024 test ===")
    print(result[cols].round(4).to_string(index=False))
    print("\n=== Model 5 minus other model, day-block bootstrap (positive = Model 5 better) ===")
    print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
