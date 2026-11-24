"""Gu-Kelly-Xiu style models on the wide factor set.

Features = cross-sectional ranks of: original factor library (|ICIR|>0.05) + 6 public factors
+ Alpha158/Alpha360-style factors passing the Harvey-Liu-Zhu |t|>3 hurdle, plus scaled raw
factors and context. Models: ridge, tuned XGBoost, LightGBM. Same 4 DATE_ID folds.
    python -m src.wide
Writes data/processed/zoo/<model>_wide_{oof,test}.parquet and reports/wide_results.csv.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .metric import weighted_sign_accuracy
from .models import rank_frame
from .research import _corr_by_group, _group_codes, cs_rank
from .zoo import OUT, load_xy

XGB_BEST = dict(learning_rate=0.03, max_depth=3, min_child_weight=10.0, subsample=0.7,
                colsample_bytree=0.9, reg_lambda=50.0, gamma=1.0)


def main():
    tr, te, Rtr, Rte, Ztr, Zte, Ctr, Cte = load_xy()
    rep = pd.read_csv("reports/a158_report.csv", index_col=0)
    keep = rep.index[rep.passes_HLZ_t3].tolist()
    A_tr = pd.read_parquet("data/processed/a158_train.parquet")[keep]
    A_te = pd.read_parquet("data/processed/a158_test.parquet")[keep]
    A_tr.index, A_te.index = tr.index, te.index
    RA_tr = rank_frame(pd.concat([tr[["DATE_ID", "COUNTRY"]], A_tr], axis=1), keep)
    RA_te = rank_frame(pd.concat([te[["DATE_ID", "COUNTRY"]], A_te], axis=1), keep)
    RA_tr.columns = RA_te.columns = [f"a_{c}" for c in RA_tr.columns]
    lin_tr, lin_te = pd.concat([Rtr, RA_tr], axis=1), pd.concat([Rte, RA_te], axis=1)
    X_tr = pd.concat([lin_tr, Ztr, Ctr], axis=1).to_numpy(np.float32)
    X_te = pd.concat([lin_te, Zte, Cte], axis=1).to_numpy(np.float32)
    print("wide features:", X_tr.shape[1], f"(old {Rtr.shape[1] + Ztr.shape[1] + Ctr.shape[1]})", flush=True)

    y = tr["y"].to_numpy()
    codes, codes_te = _group_codes(tr), _group_codes(te)
    folds = list(GroupKFold(4).split(tr, y, tr["DATE_ID"]))

    def fit_ridge(a, b):
        m = Ridge(alpha=3000.0).fit(lin_tr.iloc[a], y[a])
        return m.predict(lin_tr.iloc[b]), m.predict(lin_te)

    def fit_xgb(a, b):
        wa, wb = np.abs(y[a]) / np.abs(y[a]).mean(), np.abs(y[b]) / np.abs(y[b]).mean()
        m = xgb.XGBClassifier(n_estimators=4000, tree_method="hist", n_jobs=2,
                              early_stopping_rounds=200, eval_metric="logloss", **XGB_BEST)
        m.fit(X_tr[a], (y[a] > 0).astype(int), sample_weight=wa,
              eval_set=[(X_tr[b], (y[b] > 0).astype(int))], sample_weight_eval_set=[wb], verbose=False)
        return m.predict_proba(X_tr[b])[:, 1] - 0.5, m.predict_proba(X_te)[:, 1] - 0.5

    def fit_lgbm(a, b):
        prm = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=1000,
                   feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0,
                   verbose=-1, seed=0, num_threads=2)
        dtr = lgb.Dataset(X_tr[a], (y[a] > 0).astype(int), weight=np.abs(y[a]))
        dva = lgb.Dataset(X_tr[b], (y[b] > 0).astype(int), weight=np.abs(y[b]))
        m = lgb.train(prm, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(X_tr[b]) - 0.5, m.predict(X_te) - 0.5

    rows = {}
    for name, fn in (("ridge_wide", fit_ridge), ("xgb_wide", fit_xgb), ("lgbm_wide", fit_lgbm)):
        t0 = time.time()
        oof, test = np.zeros(len(tr)), np.zeros(len(te))
        for a, b in folds:
            pb, pt = fn(a, b)
            oof[b], test = pb, test + pt / 4
        oof = oof - pd.Series(oof).groupby(codes).transform("mean").to_numpy()
        test = test - pd.Series(test).groupby(codes_te).transform("mean").to_numpy()
        fs = [weighted_sign_accuracy(y[b], oof[b]) for _, b in folds]
        ic = _corr_by_group(cs_rank(pd.Series(oof), codes), y, codes)
        ic = ic[np.isfinite(ic)]
        rows[name] = dict(score=np.mean(fs), score_std=np.std(fs), RankIC=ic.mean(),
                          ICIR=ic.mean() / ic.std(), folds=np.round(fs, 4).tolist())
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[name].items()},
              f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame({"oof": oof}).to_parquet(OUT / f"{name}_oof.parquet")
        pd.DataFrame({"test": test}).to_parquet(OUT / f"{name}_test.parquet")
    pd.DataFrame(rows).T.to_csv("reports/wide_results.csv")


if __name__ == "__main__":
    main()
