"""Portfolio backtest on out-of-fold predictions.

Each day, in each country: enter at the cutoff, exit at the close (30-minute holding).
Weights are country-neutral (sum to 0 within every country-day) and the book has gross 1.
Since y is already the country-relative return, PnL_d = sum_i w_i * y_i.

Two weightings:
  rank  w ∝ centred cross-sectional rank of the prediction   (classic factor portfolio)
  sign  w ∝ sign(prediction - cross-sectional median)        (what the metric rewards)

Metrics: mean daily return (bp of gross), vol, annualised Sharpe (sqrt(252)), daily hit rate,
quintile returns, and break-even cost per side (entry + exit = 2 trades per day).
DATE_ID order is not calendar order, so the cumulative curve is in DATE_ID order and its
shape (drawdowns) is indicative only; Sharpe and hit rate do not depend on order.
"""
import json

import numpy as np
import pandas as pd

from .metric import weighted_sign_accuracy
from .research import _corr_by_group, _group_codes, cs_rank

P = "data/processed/"


def portfolio(df, pred_col, scheme="rank"):
    codes = _group_codes(df)
    r = cs_rank(df[pred_col], codes)
    w = np.sign(r) if scheme == "sign" else r
    w = w - pd.Series(w).groupby(codes).transform("mean").to_numpy()      # country-neutral
    gross = pd.Series(np.abs(w)).groupby(df["DATE_ID"].to_numpy()).transform("sum").to_numpy()
    w = w / gross                                                          # gross 1 per day
    pnl = pd.Series(w * df["y"].to_numpy()).groupby(df["DATE_ID"].to_numpy()).sum()
    return pnl


def stats(pnl):
    mu, sd = pnl.mean(), pnl.std()
    cum = pnl.cumsum()
    return dict(mean_bp=mu * 1e4, vol_bp=sd * 1e4, sharpe=mu / sd * np.sqrt(252),
                hit=(pnl > 0).mean(), maxdd_bp=(cum - cum.cummax()).min() * 1e4,
                breakeven_cost_bp=mu / 2 * 1e4, days=len(pnl))


def main():
    df = pd.read_parquet(P + "preds_train.parquet")
    codes = _group_codes(df)
    y = df["y"].to_numpy()

    # blends in z-space (keep magnitude information) + cross-sectional demeaning
    z = lambda c: (df[c] - df[c].mean()) / df[c].std()
    df["blend_rl"] = z("ridge") + z("lgbm")
    df["blend_all"] = z("ridge") + z("lgbm") + z("composite")
    models = ["composite", "ridge", "lgbm", "ensemble", "blend_rl", "blend_all"]
    for m in list(models):
        dm = df[m] - df.groupby(codes)[m].transform("mean")
        df[m + "_dm"] = dm
        models.append(m + "_dm")

    rows, curves = [], {}
    for m in models:
        ic = _corr_by_group(cs_rank(df[m], codes), y, codes)
        ic = ic[np.isfinite(ic)]
        row = dict(model=m, score=weighted_sign_accuracy(y, df[m].replace(0, 1e-12)),
                   RankIC=ic.mean(), ICIR=ic.mean() / ic.std())
        for sch in ("rank", "sign"):
            s = stats(portfolio(df, m, sch))
            row.update({f"{sch}_{k}": v for k, v in s.items() if k != "days"})
        q = np.clip(((cs_rank(df[m], codes) + 0.5) * 5).astype(int), 0, 4)
        qm = pd.Series(y).groupby(q).mean() * 1e4
        row.update({f"Q{i+1}_bp": qm.iloc[i] for i in range(5)})
        rows.append(row)
        curves[m] = portfolio(df, m, "rank").cumsum().mul(1e4).round(2).tolist()
    res = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 250)
    print(res[["score", "RankIC", "ICIR", "rank_mean_bp", "rank_sharpe", "rank_hit",
               "sign_sharpe", "rank_breakeven_cost_bp", "Q1_bp", "Q3_bp", "Q5_bp"]].round(4).to_string())
    res.to_csv("reports/backtest.csv")
    json.dump(curves, open("reports/curves.json", "w"))


if __name__ == "__main__":
    main()
