"""Factor library.

Each factor is a raw stock-day signal computed from the open->cutoff 5-min path. Signs are
NOT pre-flipped: the single-factor IC test decides the direction. Factors are grouped by
family for the report.

Market (country-day) quantities are equal-weighted across the stocks of the same
DATE_ID x COUNTRY, which is exactly the benchmark the target is measured against.
"""
import numpy as np
import pandas as pd

from .data import ffill_rows, price_matrix

FAMILIES = {}


def _reg(family, name, values, out):
    FAMILIES[name] = family
    out[name] = values


def _lagged_ret(lr, k):
    return np.nansum(lr[:, -k:], axis=1)


def build_factors(df: pd.DataFrame, stock_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = price_matrix(df)
    p = ffill_rows(raw)
    n_rows, n = p.shape
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(p)
        lr = np.diff(lp, axis=1)                     # (n, 95) 5-min log returns
        lr0 = np.nan_to_num(lr)

        # ---------- market path per country-day ----------
        key = df["DATE_ID"].astype(str) + "_" + df["COUNTRY"].astype(str)
        codes, uniq = pd.factorize(key)
        cnt = np.bincount(codes, minlength=len(uniq)).astype(float)
        mkt = np.zeros((len(uniq), lr.shape[1]))
        np.add.at(mkt, codes, lr0)
        mkt /= cnt[:, None]
        m = mkt[codes]                                # market return path aligned to rows

        # beta and residuals over the day (skip first 6 bars: opening noise)
        s = slice(6, None)
        mx = m[:, s] - m[:, s].mean(1, keepdims=True)
        rx = lr0[:, s] - lr0[:, s].mean(1, keepdims=True)
        beta = (mx * rx).sum(1) / ((mx ** 2).sum(1) + 1e-18)
        resid = lr0 - beta[:, None] * m

        rv = np.sqrt(np.nansum(lr ** 2, axis=1))
        rv_late = np.sqrt(np.nansum(lr[:, -24:] ** 2, axis=1))
        nb = np.sum(~np.isnan(lr), axis=1).clip(min=1)

        # ---------- 1. reversal / momentum ----------
        for k in (1, 2, 3, 6, 12, 24):
            _reg("reversal", f"ret_{k}", _lagged_ret(lr, k), out)
        _reg("momentum", "ret_day", np.nansum(lr, axis=1), out)
        _reg("momentum", "ret_open30", np.nansum(lr[:, :6], axis=1), out)
        _reg("momentum", "ret_ex_open", np.nansum(lr[:, 6:], axis=1), out)
        _reg("momentum", "ret_mid", np.nansum(lr[:, 6:-6], axis=1), out)

        # ---------- 2. residual (beta-hedged) reversal ----------
        for k in (1, 2, 3, 6, 12):
            _reg("residual", f"res_{k}", resid[:, -k:].sum(1), out)
        _reg("residual", "res_day", resid.sum(1), out)
        _reg("residual", "res_day_z", resid.sum(1) / (resid.std(1) * np.sqrt(nb) + 1e-12), out)
        for k in (1, 3, 6):
            _reg("residual", f"res_{k}_z", resid[:, -k:].sum(1) / (rv_late + 1e-12), out)

        # ---------- 3. volatility-scaled reversal ----------
        for k in (1, 2, 3, 6):
            _reg("scaled", f"z_ret_{k}", _lagged_ret(lr, k) / (rv + 1e-12), out)
        _reg("scaled", "z_ret_day", np.nansum(lr, axis=1) / (rv + 1e-12), out)

        # ---------- 4. market interaction (target is relative to country) ----------
        _reg("market", "beta", beta, out)
        for k in (1, 3, 6, 12):
            _reg("market", f"beta_x_mkt_{k}", (beta - 1) * m[:, -k:].sum(1), out)
        _reg("market", "beta_x_mkt_day", (beta - 1) * m.sum(1), out)

        # ---------- 5. volatility & distribution ----------
        _reg("volatility", "rv", rv, out)
        _reg("volatility", "rv_ratio_late", rv_late / (rv + 1e-12), out)
        _reg("volatility", "idio_vol", resid.std(1), out)
        rv2 = rv ** 2
        _reg("distribution", "rskew", np.sqrt(nb) * np.nansum(lr ** 3, 1) / rv2 ** 1.5, out)
        _reg("distribution", "rkurt", nb * np.nansum(lr ** 4, 1) / rv2 ** 2, out)
        up = np.nansum(np.where(lr > 0, lr ** 2, 0), 1)
        _reg("distribution", "rsj", (2 * up - rv2) / rv2, out)
        bv = (np.pi / 2) * np.nansum(np.abs(lr[:, 1:]) * np.abs(lr[:, :-1]), 1)
        _reg("distribution", "jump", (rv2 - bv) / rv2, out)
        _reg("distribution", "max_ret", np.nanmax(lr, 1), out)
        _reg("distribution", "min_ret", np.nanmin(lr, 1), out)

        # ---------- 6. path shape ----------
        _reg("path", "path_eff", np.nansum(lr, 1) / np.nansum(np.abs(lr), 1), out)
        _reg("path", "path_eff_late", np.nansum(lr[:, -24:], 1) / np.nansum(np.abs(lr[:, -24:]), 1), out)
        _reg("path", "dist_high", p[:, -1] / np.nanmax(p, 1) - 1, out)
        _reg("path", "dist_low", p[:, -1] / np.nanmin(p, 1) - 1, out)
        _reg("path", "range_pos", (p[:, -1] - np.nanmin(p, 1)) / (np.nanmax(p, 1) - np.nanmin(p, 1)), out)
        _reg("path", "vs_twap", p[:, -1] / np.nanmean(p, 1) - 1, out)
        _reg("path", "vs_twap_12", p[:, -1] / np.nanmean(p[:, -12:], 1) - 1, out)
        hi = np.nanargmax(np.nan_to_num(p, nan=-np.inf), 1)
        lo = np.nanargmin(np.nan_to_num(p, nan=np.inf), 1)
        _reg("path", "bars_since_high", n - 1 - hi, out)
        _reg("path", "bars_since_low", n - 1 - lo, out)
        a, b = rx[:, 1:], rx[:, :-1]
        _reg("path", "ac1", (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)), out)

        # ---------- 7. data quality / session ----------
        _reg("session", "n_missing", np.isnan(raw).sum(1), out)

    f = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)

    # ---------- 8. stock-level (uses train-period features only, no target) ----------
    if stock_stats is not None:
        f["stock_rv"] = df["STOCK_ID"].map(stock_stats["rv"]).to_numpy()
        f["rv_vs_stock"] = f["rv"] / f["stock_rv"]
        FAMILIES["stock_rv"] = "stock"
        FAMILIES["rv_vs_stock"] = "stock"
    return f


def context(df: pd.DataFrame) -> pd.DataFrame:
    """Non-factor conditioning variables for the models."""
    c = pd.DataFrame(index=df.index)
    c["country_code"] = df["COUNTRY"].astype("category").cat.codes
    c["summer"] = (df["CLOSE_TIME"] == "15:30").astype(int)
    c["half_day"] = (df["CLOSE_TIME"] == "12:30").astype(int)
    c["cs_size"] = df.groupby(["DATE_ID", "COUNTRY"])["ID"].transform("size")
    return c
