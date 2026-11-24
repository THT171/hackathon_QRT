"""Wide factor set, after Qlib Alpha158 (price part) + a raw return path (Alpha360 / Gu-Kelly-Xiu).

Computed on 5-min bars, windows W = (3, 6, 12, 24, 48) bars, "close" = last price before the
cutoff. Volume factors of Alpha158 are dropped (no volume). Then:
  * single-factor IC on every factor (country-day cross-sections)
  * Harvey-Liu-Zhu (2016) hurdle: keep |t| > 3
  * python -m src.alpha158   -> data/processed/a158_{train,test}.parquet, reports/a158_report.csv
"""
import numpy as np
import pandas as pd

from .data import ffill_rows, load, price_matrix
from .public_factors import _bfill_first
from .research import factor_report

W = (3, 6, 12, 24, 48)


def build_a158(df):
    p = _bfill_first(ffill_rows(price_matrix(df)))
    n = p.shape[1]
    c = p[:, -1]
    lr = np.diff(np.log(p), axis=1)
    f = {}
    o, hi, lo = p[:, 0], p.max(1), p.min(1)
    # K-line features on the whole session (open = first price, close = cutoff price)
    f["KMID"] = (c - o) / o
    f["KLEN"] = (hi - lo) / o
    f["KMID2"] = (c - o) / (hi - lo + 1e-12)
    f["KUP"] = (hi - np.maximum(o, c)) / o
    f["KUP2"] = (hi - np.maximum(o, c)) / (hi - lo + 1e-12)
    f["KLOW"] = (np.minimum(o, c) - lo) / o
    f["KLOW2"] = (np.minimum(o, c) - lo) / (hi - lo + 1e-12)
    f["KSFT"] = (2 * c - hi - lo) / o
    f["KSFT2"] = (2 * c - hi - lo) / (hi - lo + 1e-12)
    for d in W:
        x = p[:, -d:]
        r = lr[:, -d:]
        t = np.arange(d) - (d - 1) / 2
        lx = np.log(x)
        slope = (lx * t).sum(1) / (t ** 2).sum()
        fit = lx.mean(1, keepdims=True) + slope[:, None] * t
        ss_res = ((lx - fit) ** 2).sum(1)
        ss_tot = ((lx - lx.mean(1, keepdims=True)) ** 2).sum(1)
        f[f"ROC{d}"] = p[:, -d - 1] / c
        f[f"MA{d}"] = x.mean(1) / c
        f[f"STD{d}"] = x.std(1) / c
        f[f"BETA{d}"] = slope
        f[f"RSQR{d}"] = 1 - ss_res / (ss_tot + 1e-18)
        f[f"RESI{d}"] = lx[:, -1] - fit[:, -1]
        f[f"MAX{d}"] = x.max(1) / c
        f[f"MIN{d}"] = x.min(1) / c
        f[f"QTLU{d}"] = np.quantile(x, 0.8, axis=1) / c
        f[f"QTLD{d}"] = np.quantile(x, 0.2, axis=1) / c
        f[f"RANK{d}"] = (x <= c[:, None]).mean(1)
        f[f"RSV{d}"] = (c - x.min(1)) / (x.max(1) - x.min(1) + 1e-12)
        imax, imin = x.argmax(1), x.argmin(1)
        f[f"IMAX{d}"] = (d - 1 - imax) / d
        f[f"IMIN{d}"] = (d - 1 - imin) / d
        f[f"IMXD{d}"] = (imax - imin) / d
        f[f"CNTP{d}"] = (r > 0).mean(1)
        f[f"CNTN{d}"] = (r < 0).mean(1)
        f[f"CNTD{d}"] = f[f"CNTP{d}"] - f[f"CNTN{d}"]
        up, dn = np.clip(r, 0, None).sum(1), -np.clip(r, None, 0).sum(1)
        f[f"SUMP{d}"] = up / (up + dn + 1e-12)
        f[f"SUMN{d}"] = dn / (up + dn + 1e-12)
        f[f"SUMD{d}"] = (up - dn) / (up + dn + 1e-12)
    # Alpha360-style raw path: last 24 returns scaled by the day's volatility
    vol = np.sqrt((lr ** 2).sum(1)) + 1e-12
    for k in range(1, 25):
        f[f"PATH{k}"] = lr[:, -k] / vol
    # 30-min block returns across the day (16 blocks ending at the cutoff), vol-scaled
    for j in range(1, 16):
        f[f"BLK{j}"] = lr[:, -6 * j:-6 * (j - 1) or None].sum(1) / vol
    out = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32)


if __name__ == "__main__":
    train, test = load()
    for name, df in (("train", train), ("test", test)):
        build_a158(df).reset_index(drop=True).to_parquet(f"data/processed/a158_{name}.parquet")
    base = pd.read_parquet("data/processed/factors_train.parquet")
    F = pd.read_parquet("data/processed/a158_train.parquet")
    fam = {c: ("a360_path" if c.startswith(("PATH", "BLK")) else "a158") for c in F.columns}
    rep, _ = factor_report(F, base.y.to_numpy(), base, base.stock_rv.to_numpy(), fam)
    rep["passes_HLZ_t3"] = rep["t"].abs() > 3
    rep.to_csv("reports/a158_report.csv")
    print(F.shape[1], "factors;", int(rep.passes_HLZ_t3.sum()), "pass |t|>3;",
          int((rep.t.abs() > 2).sum()), "pass |t|>2")
    pd.set_option("display.width", 200)
    print(rep[["family", "RankIC", "ICIR", "t", "LS_bp", "score"]].head(25).round(4).to_string())
