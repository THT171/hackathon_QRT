"""Public / published factors, adapted to 5-min prices with no volume and no previous day.

Sources (adapted, see README):
  gao_*      Gao, Han, Li & Zhou (2018 JFE) market intraday momentum: first half-hour and
             second-to-last half-hour market returns predict the last half-hour. Market-level,
             so in the cross-section it enters through beta.
  eod_*      Baltussen, Da, Lammers & Martens (2025) end-of-day reversal: return up to 30 min
             before the cutoff-window (skip the second-to-last half-hour), stronger for losers.
  a101_*     Kakushadze (2016) "101 Formulaic Alphas", bar-level versions of #101, #9-like
             sign-consistency, ts_rank.
  ta_*       Technical indicators: RSI(14), Bollinger %B(20), MACD histogram, stochastic %K.
  cn_*       Chinese broker high-frequency factors: ideal reversal (Kaiyuan W-cut, split by bar
             amplitude), ideal amplitude, upper/lower shadow, max drawdown / run-up, surge moments
             (Founder "moderate risk"), variance ratio, time-centre of price mass.
  peer_*     Connected stocks / lead-lag (Cohen-Frazzini 2008; Hou 2007): peers = 10 stocks with
             the highest 30-min return correlation in the training period.
"""
import numpy as np
import pandas as pd

from .data import ffill_rows, price_matrix

PUB_FAMILY = {}


def _add(fam, name, v, out):
    PUB_FAMILY[name] = fam
    out[name] = v


def _ema(x, span):
    a = 2 / (span + 1)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = np.where(np.isnan(x[:, t]), out[:, t - 1], a * x[:, t] + (1 - a) * out[:, t - 1])
    return out


def _bfill_first(p):
    """Fill leading NaNs with the first valid price (so short days get flat early bars)."""
    first = np.argmax(~np.isnan(p), 1)
    fv = p[np.arange(len(p)), first]
    return np.where(np.isnan(p), fv[:, None], p)


def _market(lr0, codes, ncodes):
    cnt = np.bincount(codes, minlength=ncodes).astype(float)
    m = np.zeros((ncodes, lr0.shape[1]))
    np.add.at(m, codes, lr0)
    return (m / cnt[:, None])[codes]


def build_public(df: pd.DataFrame, peers: dict | None = None) -> pd.DataFrame:
    p = _bfill_first(ffill_rows(price_matrix(df)))
    n = p.shape[1]
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(p), axis=1)
        lr0 = np.nan_to_num(lr)
        key = df["DATE_ID"].astype(str) + "_" + df["COUNTRY"].astype(str)
        codes, uniq = pd.factorize(key)
        m = _market(lr0, codes, len(uniq))
        s = slice(6, None)
        mx = m[:, s] - m[:, s].mean(1, keepdims=True)
        rx = lr0[:, s] - lr0[:, s].mean(1, keepdims=True)
        beta = (mx * rx).sum(1) / ((mx ** 2).sum(1) + 1e-18)
        rv = np.sqrt((lr0 ** 2).sum(1)) + 1e-12

        # ---- Gao et al. market intraday momentum (via beta) ----
        mkt_fh, mkt_slh = m[:, :6].sum(1), m[:, -6:].sum(1)
        _add("gao", "gao_beta_x_fh", (beta - 1) * mkt_fh, out)
        _add("gao", "gao_beta_x_slh", (beta - 1) * mkt_slh, out)
        _add("gao", "gao_beta_x_fh_slh", (beta - 1) * (mkt_fh + mkt_slh), out)
        _add("gao", "gao_stock_fh_slh", lr0[:, :6].sum(1) + lr0[:, -6:].sum(1), out)

        # ---- Baltussen-Da end-of-day reversal ----
        rod = lr0[:, :-6].sum(1)                      # open -> 30 min before cutoff (skip SLH)
        rod_rel = rod - pd.Series(rod).groupby(codes).transform("mean").to_numpy()
        _add("eod", "eod_rod", rod, out)
        _add("eod", "eod_rod_loser", np.minimum(rod_rel, 0), out)
        _add("eod", "eod_rod_winner", np.maximum(rod_rel, 0), out)
        _add("eod", "eod_rod_z", rod / rv, out)

        # ---- Alpha101 adaptations ----
        o, c = p[:, 0], p[:, -1]
        hi, lo = p.max(1), p.min(1)
        _add("a101", "a101_body_day", (c - o) / (hi - lo + 1e-9 * c), out)
        o6, h6, l6 = p[:, -7], p[:, -7:].max(1), p[:, -7:].min(1)
        _add("a101", "a101_body_30m", (c - o6) / (h6 - l6 + 1e-9 * c), out)
        cons = np.sign(lr0[:, -5:]).mean(1)
        _add("a101", "a101_9_signcons", np.where(np.abs(cons) == 1, lr0[:, -1], -lr0[:, -1]), out)
        _add("a101", "a101_tsrank_12", (p[:, -12:] <= c[:, None]).mean(1), out)

        # ---- technical indicators ----
        d = lr0[:, -14:]
        up, dn = np.clip(d, 0, None).sum(1), -np.clip(d, None, 0).sum(1)
        _add("ta", "ta_rsi14", up / (up + dn + 1e-12) - 0.5, out)
        w = p[:, -20:]
        _add("ta", "ta_boll_b", (c - w.mean(1)) / (2 * w.std(1) + 1e-9 * c), out)
        macd = _ema(p, 12) - _ema(p, 26)
        sig = _ema(macd, 9)
        _add("ta", "ta_macd_hist", (macd[:, -1] - sig[:, -1]) / c, out)
        h14, l14 = p[:, -14:].max(1), p[:, -14:].min(1)
        _add("ta", "ta_stoch_k", (c - l14) / (h14 - l14 + 1e-9 * c) - 0.5, out)

        # ---- Chinese broker high-frequency factors ----
        a = np.abs(lr0)
        order = np.argsort(a, axis=1)
        k = int(0.3 * lr0.shape[1])
        rows = np.arange(len(p))[:, None]
        big, small = order[:, -k:], order[:, :k]
        _add("cn", "cn_ideal_reversal", lr0[rows, big].sum(1) - lr0[rows, small].sum(1), out)
        _add("cn", "cn_ret_big_bars", lr0[rows, big].sum(1), out)
        _add("cn", "cn_ret_small_bars", lr0[rows, small].sum(1), out)
        pq = p[:, 1:]
        hi_q, lo_q = np.quantile(pq, 0.75, axis=1), np.quantile(pq, 0.25, axis=1)
        amp_hi = np.where(pq >= hi_q[:, None], a, np.nan)
        amp_lo = np.where(pq <= lo_q[:, None], a, np.nan)
        _add("cn", "cn_ideal_amplitude", (np.nanmean(amp_hi, 1) - np.nanmean(amp_lo, 1)) / rv, out)
        _add("cn", "cn_upper_shadow", (hi - np.maximum(o, c)) / c, out)
        _add("cn", "cn_lower_shadow", (np.minimum(o, c) - lo) / c, out)
        h12, l12 = p[:, -12:].max(1), p[:, -12:].min(1)
        o12 = p[:, -12]
        _add("cn", "cn_upper_shadow_1h", (h12 - np.maximum(o12, c)) / c, out)
        _add("cn", "cn_lower_shadow_1h", (np.minimum(o12, c) - l12) / c, out)
        lp = np.log(p)
        _add("cn", "cn_max_drawdown", (lp - np.maximum.accumulate(lp, 1)).min(1), out)
        _add("cn", "cn_max_runup", (lp - np.minimum.accumulate(lp, 1)).max(1), out)
        thr = a.mean(1, keepdims=True) + a.std(1, keepdims=True)
        surge = a > thr
        nxt = np.zeros_like(lr0)
        nxt[:, :-1] = lr0[:, 1:]
        _add("cn", "cn_post_surge_ret", np.where(surge[:, :-1], nxt[:, :-1], 0).sum(1) / (surge[:, :-1].sum(1) + 1), out)
        _add("cn", "cn_surge_ret", np.where(surge, lr0, 0).sum(1), out)
        _add("cn", "cn_surges_late", surge[:, -12:].sum(1), out)
        r6 = lr0[:, : (lr0.shape[1] // 6) * 6].reshape(len(p), -1, 6).sum(2)
        _add("cn", "cn_var_ratio6", r6.var(1) / (6 * lr0.var(1) + 1e-18), out)
        t = np.arange(n)
        wgt = p / p.sum(1, keepdims=True)
        _add("cn", "cn_price_time_centre", (wgt * t).sum(1) / (n - 1) - 0.5, out)

    f = pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan)

    # ---- connected-stock peers ----
    if peers is not None:
        res6 = (lr0[:, -6:] - beta[:, None] * m[:, -6:]).sum(1)
        resd = (lr0 - beta[:, None] * m).sum(1)
        for name, v in (("peer_res6", res6), ("peer_resday", resd)):
            M = pd.DataFrame({"d": df["DATE_ID"].to_numpy(), "s": df["STOCK_ID"].to_numpy(), "v": v}) \
                .pivot(index="d", columns="s", values="v")
            W = peers["W"].reindex(index=M.columns, columns=M.columns).fillna(0).to_numpy()
            X = M.to_numpy()
            num = np.nan_to_num(X) @ W.T
            den = (~np.isnan(X)).astype(float) @ W.T
            P = pd.DataFrame(num / np.where(den > 0, den, np.nan), index=M.index, columns=M.columns)
            stacked = P.stack()
            idx = pd.MultiIndex.from_arrays([df["DATE_ID"], df["STOCK_ID"]])
            pv = stacked.reindex(idx).to_numpy()
            _add("peer", name, pv, f)
            _add("peer", name + "_gap", pv - v, f)   # own lag vs peers: catch-up
    return f


def fit_peers(train: pd.DataFrame, k: int = 10) -> dict:
    """Peers from correlation of 30-min returns over the training period (no target used)."""
    p = _bfill_first(ffill_rows(price_matrix(train)))
    lr = np.diff(np.log(p), axis=1)
    r30 = np.nan_to_num(lr[:, -90:]).reshape(len(p), 15, 6).sum(2)          # 15 half-hours
    long = pd.DataFrame(r30)
    long["d"], long["s"] = train["DATE_ID"].to_numpy(), train["STOCK_ID"].to_numpy()
    long = long.melt(id_vars=["d", "s"], var_name="slot", value_name="r")
    mat = long.pivot_table(index=["d", "slot"], columns="s", values="r")
    mat = mat - mat.mean(1).to_numpy()[:, None]                              # remove market
    C = mat.corr(min_periods=300)
    Cv = C.to_numpy().copy()
    np.fill_diagonal(Cv, np.nan)
    C = pd.DataFrame(Cv, index=C.index, columns=C.columns)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for s in C.index:
        top = C.loc[s].nlargest(k)
        W.loc[s, top.index] = np.clip(top.to_numpy(), 0, None)
    return {"W": W, "C": C}
