"""Single-factor evaluation, computed on each DATE_ID x COUNTRY cross-section.

For every factor:
  IC        Pearson corr(factor, y) per cross-section (factor winsorised at 1/99 pct)
  RankIC    Spearman corr per cross-section
  wIC       |y|-free weighted IC: Pearson corr weighted by the stock's train-period volatility,
            closer to what the competition metric rewards (big movers count more)
  ICIR      mean(RankIC) / std(RankIC);  t = ICIR * sqrt(#cross-sections)
  hit       share of cross-sections with RankIC of the expected sign
  Q1..Q5    mean target per within-cross-section quintile; LS = Q5 - Q1 (bp)
  score     competition metric of sign(direction * demeaned factor)
"""
import numpy as np
import pandas as pd

from .metric import weighted_sign_accuracy


def _group_codes(df):
    return pd.factorize(df["DATE_ID"].astype(str) + "_" + df["COUNTRY"].astype(str))[0]


def _demean(x, codes, w=None):
    w = np.ones_like(x) if w is None else w
    s = np.bincount(codes, weights=x * w)
    c = np.bincount(codes, weights=w)
    return x - (s / c)[codes]


def _corr_by_group(x, y, codes, w=None):
    """Weighted Pearson correlation of x and y within each group. NaNs in x -> group mean."""
    w = np.ones_like(y) if w is None else w
    xd, yd = _demean(x, codes, w), _demean(y, codes, w)
    sxy = np.bincount(codes, weights=w * xd * yd)
    sxx = np.bincount(codes, weights=w * xd * xd)
    syy = np.bincount(codes, weights=w * yd * yd)
    return sxy / np.sqrt(sxx * syy + 1e-30)


def cs_rank(x: pd.Series, codes) -> np.ndarray:
    """Percentile rank within cross-section, centred to [-0.5, 0.5]; NaN -> 0."""
    r = pd.Series(x.to_numpy()).groupby(codes).rank(pct=True).to_numpy() - 0.5
    return np.nan_to_num(r)


def winsor(x, q=0.01):
    lo, hi = np.nanquantile(x, [q, 1 - q])
    return np.clip(x, lo, hi)


def factor_report(F: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, w: np.ndarray, families: dict):
    codes = _group_codes(meta)
    rows, rank_ics = [], {}
    for name in F.columns:
        x = F[name].to_numpy(dtype=float)
        if np.isnan(x).all() or np.nanstd(x) == 0:
            continue
        xw = winsor(x)
        xw = np.where(np.isnan(xw), np.nan, xw)
        fill = pd.Series(xw).groupby(codes).transform("mean").fillna(0).to_numpy()
        xw = np.where(np.isnan(xw), fill, xw)
        r = cs_rank(F[name], codes)
        ic = _corr_by_group(xw, y, codes)
        ric = _corr_by_group(r, y, codes)
        wic = _corr_by_group(xw, y, codes, w)
        ric = ric[np.isfinite(ric)]
        rank_ics[name] = pd.Series(_corr_by_group(r, y, codes))
        direction = np.sign(ric.mean()) or 1.0
        q = np.clip(((r + 0.5) * 5).astype(int), 0, 4)  # quintiles within cross-section
        qm = pd.Series(y).groupby(q).mean().reindex(range(5)).to_numpy() * 1e4
        sig = direction * _demean(xw, codes)
        rows.append(dict(
            factor=name, family=families.get(name, "other"),
            IC=np.nanmean(ic), RankIC=ric.mean(), wIC=np.nanmean(wic),
            ICIR=ric.mean() / ric.std(), t=ric.mean() / ric.std() * np.sqrt(len(ric)),
            hit=(np.sign(ric) == direction).mean(),
            Q1=qm[0], Q2=qm[1], Q3=qm[2], Q4=qm[3], Q5=qm[4], LS_bp=qm[4] - qm[0],
            score=weighted_sign_accuracy(y, np.where(sig == 0, 1e-12, sig)),
        ))
    rep = pd.DataFrame(rows).set_index("factor")
    rep["absICIR"] = rep["ICIR"].abs()
    return rep.sort_values("absICIR", ascending=False), pd.DataFrame(rank_ics)
