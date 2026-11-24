"""Intraday features from the 5-min price path (open -> cutoff).

All features use only price ratios (price level is arbitrary). Every feature also gets a
country-day demeaned version, because the target is relative to the country mean.
"""
import numpy as np
import pandas as pd

from .data import ffill_rows, price_matrix


def _ret(p, a, b):
    """Return from bar index a to bar index b (negative indices allowed)."""
    return p[:, b] / p[:, a] - 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    p = ffill_rows(price_matrix(df))
    n = p.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(p), axis=1)  # 5-min log returns
        f = {}

        # --- recent returns (short-term reversal / momentum) ---
        for k in (1, 2, 3, 6, 12, 24):
            f[f"ret_last_{k}"] = _ret(p, n - 1 - k, n - 1)
        first = np.argmax(~np.isnan(p), axis=1)
        p_first = p[np.arange(len(p)), first]
        f["ret_day"] = p[:, -1] / p_first - 1
        f["ret_first_30m"] = p[np.arange(len(p)), np.minimum(first + 6, n - 1)] / p_first - 1
        f["ret_mid"] = p[:, -7] / p[np.arange(len(p)), np.minimum(first + 6, n - 1)] - 1

        # --- volatility / distribution ---
        rv = np.nansum(lr ** 2, axis=1)
        nb = np.sum(~np.isnan(lr), axis=1).clip(min=1)
        f["rv"] = np.sqrt(rv)
        f["rv_last_12"] = np.sqrt(np.nansum(lr[:, -12:] ** 2, axis=1))
        f["rskew"] = np.sqrt(nb) * np.nansum(lr ** 3, axis=1) / rv ** 1.5
        f["rkurt"] = nb * np.nansum(lr ** 4, axis=1) / rv ** 2
        up = np.nansum(np.where(lr > 0, lr ** 2, 0), axis=1)
        f["rsj"] = (2 * up - rv) / rv
        bv = (np.pi / 2) * np.nansum(np.abs(lr[:, 1:]) * np.abs(lr[:, :-1]), axis=1)
        f["jump"] = (rv - bv) / rv
        f["max_ret"] = np.nanmax(lr, axis=1)
        f["min_ret"] = np.nanmin(lr, axis=1)

        # --- path shape ---
        f["path_eff"] = (np.log(p[:, -1]) - np.log(p_first)) / np.nansum(np.abs(lr), axis=1)
        f["dist_high"] = p[:, -1] / np.nanmax(p, axis=1) - 1
        f["dist_low"] = p[:, -1] / np.nanmin(p, axis=1) - 1
        f["vs_twap"] = p[:, -1] / np.nanmean(p, axis=1) - 1
        f["vs_twap_last12"] = p[:, -1] / np.nanmean(p[:, -12:], axis=1) - 1
        f["ac1"] = _row_autocorr(lr)
        f["n_missing"] = np.isnan(price_matrix(df)).sum(axis=1)

        # normalised versions (return in units of the day's volatility)
        for k in (1, 3, 12):
            f[f"z_ret_last_{k}"] = f[f"ret_last_{k}"] / (f["rv"] + 1e-12)
        f["z_ret_day"] = f["ret_day"] / (f["rv"] + 1e-12)

    feats = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)

    # country-day relative features (target is relative to country mean)
    key = [df["DATE_ID"], df["COUNTRY"]]
    base = list(feats.columns)
    g = feats.groupby(key)
    rel = feats[base] - g[base].transform("mean")
    rel.columns = [f"{c}_rel" for c in base]
    rank = g[base].rank(pct=True) - 0.5
    rank.columns = [f"{c}_rank" for c in base]

    # market (country-day) context
    ctx = g[["ret_day", "ret_last_6", "rv"]].transform("mean")
    ctx.columns = [f"cty_{c}" for c in ctx.columns]
    ctx["cty_size"] = g["ret_day"].transform("size")

    out = pd.concat([feats, rel, rank, ctx], axis=1)
    out["country_code"] = df["COUNTRY"].astype("category").cat.codes
    return out


def _row_autocorr(x):
    a, b = x[:, 1:], x[:, :-1]
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    return np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a ** 2, axis=1) * np.nansum(b ** 2, axis=1))
