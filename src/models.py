"""Model comparison on the factor library, 4-fold CV grouped by DATE_ID.

Models
  composite  ICIR-weighted sum of decorrelated factor ranks (weights estimated in-fold)
  ridge      ridge regression of y on cross-sectional ranks of all candidate factors
  lgbm       LightGBM binary on sign(y) weighted by |y|; ranks + raw scaled factors + context
  ensemble   average of the three models' cross-sectional ranks

Writes OOF + test predictions to data/processed/preds_{train,test}.parquet.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .metric import weighted_sign_accuracy
from .research import _corr_by_group, _group_codes, cs_rank

P = "data/processed/"
META = ["ID", "DATE_ID", "STOCK_ID", "COUNTRY", "CLOSE_TIME", "y"]
CTX = ["country_code", "summer", "half_day", "cs_size", "stock_rv"]

PUBLIC_ADD = ["a101_9_signcons", "cn_upper_shadow", "cn_lower_shadow", "ta_boll_b",
              "peer_res6_gap", "ta_stoch_k"]

LGB = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=1000,
           feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0,
           verbose=-1, seed=0, num_threads=8)


def rank_frame(df, cols):
    codes = _group_codes(df)
    return pd.DataFrame({f"r_{c}": cs_rank(df[c], codes) for c in cols}, index=df.index)


def ic_stats(pred, y, codes):
    ic = _corr_by_group(cs_rank(pd.Series(pred), codes), y, codes)
    ic = ic[np.isfinite(ic)]
    return ic.mean(), ic.mean() / ic.std()


def fit_composite(R, y, codes, sel):
    ics = {c: _corr_by_group(R[f"r_{c}"].to_numpy(), y, codes) for c in sel}
    w = {c: np.nanmean(v) / np.nanstd(v) for c, v in ics.items()}  # ICIR weights
    return w


def predict_composite(R, w):
    return sum(R[f"r_{c}"].to_numpy() * wi for c, wi in w.items())


def main():
    tr = pd.read_parquet(P + "factors_train.parquet")
    te = pd.read_parquet(P + "factors_test.parquet")
    rep = pd.read_csv("reports/factor_report.csv", index_col=0)
    cand = rep[rep.absICIR > 0.05].index.tolist()
    sel = pd.read_csv("reports/selected_factors.csv").iloc[:, 0].tolist()
    # public factors with significant incremental IC (partial |t| > 2, reports/public_incremental.csv)
    pub_cols = PUBLIC_ADD
    if pub_cols:
        tr = pd.concat([tr, pd.read_parquet(P + "public_train.parquet")[pub_cols]], axis=1)
        te = pd.concat([te, pd.read_parquet(P + "public_test.parquet")[pub_cols]], axis=1)
        cand, sel = cand + pub_cols, sel + ["a101_9_signcons", "cn_upper_shadow", "cn_lower_shadow", "peer_res6_gap"]

    Rtr, Rte = rank_frame(tr, cand), rank_frame(te, cand)
    raw_cols = [c for c in cand if c.endswith("_z") or c.startswith("z_") or c in
                ("path_eff", "bars_since_high", "bars_since_low", "rsj", "beta", "rv")] + \
               ["cn_upper_shadow", "cn_lower_shadow", "ta_stoch_k"]
    Xtr = pd.concat([Rtr, tr[raw_cols + CTX]], axis=1)
    Xte = pd.concat([Rte, te[raw_cols + CTX]], axis=1)

    y = tr["y"].to_numpy()
    codes_tr, codes_te = _group_codes(tr), _group_codes(te)
    folds = list(GroupKFold(4).split(tr, y, tr["DATE_ID"]))

    oof = {m: np.zeros(len(tr)) for m in ("composite", "ridge", "lgbm")}
    iters = []
    for k, (a, b) in enumerate(folds):
        w = fit_composite(Rtr.iloc[a], y[a], codes_tr[a], sel)
        oof["composite"][b] = predict_composite(Rtr.iloc[b], w)

        rg = Ridge(alpha=1000.0).fit(Rtr.iloc[a], y[a])
        oof["ridge"][b] = rg.predict(Rtr.iloc[b])

        dtr = lgb.Dataset(Xtr.iloc[a], (y[a] > 0).astype(int), weight=np.abs(y[a]))
        dva = lgb.Dataset(Xtr.iloc[b], (y[b] > 0).astype(int), weight=np.abs(y[b]))
        m = lgb.train(LGB, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof["lgbm"][b] = m.predict(Xtr.iloc[b]) - 0.5
        iters.append(m.best_iteration)

    def ens(preds, codes):
        return sum(cs_rank(pd.Series(p), codes) for p in preds) / len(preds)

    oof["ensemble"] = ens([oof["composite"], oof["ridge"], oof["lgbm"]], codes_tr)

    rows = []
    for name, p in oof.items():
        fs = [weighted_sign_accuracy(y[b], p[b]) for _, b in folds]
        ic, icir = ic_stats(p, y, codes_tr)
        rows.append(dict(model=name, score=np.mean(fs), score_std=np.std(fs), RankIC=ic, ICIR=icir,
                         folds=np.round(fs, 4).tolist()))
    res = pd.DataFrame(rows).set_index("model")
    print(res.to_string())
    res.to_csv("reports/model_cv.csv")

    # ---- refit on all training data, predict test ----
    w = fit_composite(Rtr, y, codes_tr, sel)
    print("composite ICIR weights:", {k: round(v, 3) for k, v in w.items()})
    pd.Series(w).to_csv("reports/composite_weights.csv")
    rg = Ridge(alpha=1000.0).fit(Rtr, y)
    pd.Series(rg.coef_, index=Rtr.columns).sort_values().to_csv("reports/ridge_coefs.csv")
    m = lgb.train(LGB, lgb.Dataset(Xtr, (y > 0).astype(int), weight=np.abs(y)),
                  int(np.mean(iters) * 1.1))
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    imp.to_csv("reports/lgbm_importance.csv")
    test = {"composite": predict_composite(Rte, w), "ridge": rg.predict(Rte),
            "lgbm": m.predict(Xte) - 0.5}
    test["ensemble"] = ens([test["composite"], test["ridge"], test["lgbm"]], codes_te)

    pd.DataFrame({"ID": tr["ID"], "DATE_ID": tr["DATE_ID"], "COUNTRY": tr["COUNTRY"], "y": y,
                  **oof}).to_parquet(P + "preds_train.parquet")
    pd.DataFrame({"ID": te["ID"], "DATE_ID": te["DATE_ID"], "COUNTRY": te["COUNTRY"],
                  **test}).to_parquet(P + "preds_test.parquet")


if __name__ == "__main__":
    main()
