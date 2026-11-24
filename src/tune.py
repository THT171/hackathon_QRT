"""Random hyper-parameter search for the top-2 single models (XGBoost, ExtraTrees).

    python -m src.tune xgb 24
    python -m src.tune et 14

Same 4 DATE_ID folds as everywhere else. Each trial logs score (mean/std over folds), RankIC
and ICIR to reports/tune_<model>.csv. The best trial's OOF and fold-averaged test predictions
are saved as data/processed/zoo/<model>_tuned_{oof,test}.parquet.
Caveat: selecting on the same CV folds is optimistic; prefer configs that are stable across folds.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metric import weighted_sign_accuracy
from .research import _corr_by_group, _group_codes, cs_rank
from .zoo import OUT, load_xy

rng = np.random.default_rng(42)


def sample(name):
    if name == "xgb":
        return dict(learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
                    max_depth=int(rng.choice([2, 3, 4, 5, 6])),
                    min_child_weight=float(rng.choice([10, 50, 200, 500, 1000])),
                    subsample=float(rng.choice([0.5, 0.7, 0.9])),
                    colsample_bytree=float(rng.choice([0.3, 0.5, 0.7, 0.9])),
                    reg_lambda=float(rng.choice([1, 10, 50, 200])),
                    gamma=float(rng.choice([0, 0.1, 1.0])))
    if name == "et":
        return dict(max_depth=int(rng.choice([6, 8, 10, 12, 16])),
                    min_samples_leaf=int(rng.choice([100, 200, 400, 800, 1500])),
                    max_features=float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.7])),
                    n_estimators=int(rng.choice([100, 150])))
    raise ValueError(name)


def fit_predict(name, prm, Xa, ya, Xb, yb, Xt):
    if name == "xgb":
        import xgboost as xgb
        wa, wb = np.abs(ya) / np.abs(ya).mean(), np.abs(yb) / np.abs(yb).mean()
        m = xgb.XGBClassifier(n_estimators=4000, tree_method="hist", n_jobs=2,
                              early_stopping_rounds=200, eval_metric="logloss", **prm)
        m.fit(Xa, (ya > 0).astype(int), sample_weight=wa,
              eval_set=[(Xb, (yb > 0).astype(int))], sample_weight_eval_set=[wb], verbose=False)
        return m.predict_proba(Xb)[:, 1] - 0.5, m.predict_proba(Xt)[:, 1] - 0.5, m.best_iteration
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(n_jobs=2, random_state=0, **prm).fit(Xa, ya)
    return m.predict(Xb), m.predict(Xt), prm["n_estimators"]


def features(name, data):
    tr, te, Rtr, Rte, Ztr, Zte, Ctr, Cte = data
    if name == "et":
        return Rtr.to_numpy(np.float32), Rte.to_numpy(np.float32)
    return (pd.concat([Rtr, Ztr, Ctr], axis=1).to_numpy(np.float32),
            pd.concat([Rte, Zte, Cte], axis=1).to_numpy(np.float32))


def demean(p, codes):
    return p - pd.Series(p).groupby(codes).transform("mean").to_numpy()


def main(name, n_trials):
    data = load_xy()
    tr, te = data[0], data[1]
    Xtr, Xte = features(name, data)
    y = tr["y"].to_numpy()
    codes, codes_te = _group_codes(tr), _group_codes(te)
    folds = list(GroupKFold(4).split(tr, y, tr["DATE_ID"]))
    log, best = [], (-1, None, None, None)
    for t in range(n_trials):
        prm = sample(name)
        t0 = time.time()
        oof, test, its = np.zeros(len(tr)), np.zeros(len(te)), []
        for a, b in folds:
            pb, pt, it = fit_predict(name, prm, Xtr[a], y[a], Xtr[b], y[b], Xte)
            oof[b], test = pb, test + pt / 4
            its.append(it)
        oof = demean(oof, codes)
        fs = [weighted_sign_accuracy(y[b], oof[b]) for _, b in folds]
        ic = _corr_by_group(cs_rank(pd.Series(oof), codes), y, codes)
        ic = ic[np.isfinite(ic)]
        row = dict(trial=t, score=np.mean(fs), score_std=np.std(fs), worst_fold=np.min(fs),
                   RankIC=ic.mean(), ICIR=ic.mean() / ic.std(), iters=int(np.mean(its)),
                   secs=round(time.time() - t0), **prm)
        log.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(log).sort_values("score", ascending=False).to_csv(f"reports/tune_{name}.csv", index=False)
        if row["score"] > best[0]:
            best = (row["score"], prm, oof, demean(test, codes_te))
    pd.DataFrame({"oof": best[2]}).to_parquet(OUT / f"{name}_tuned_oof.parquet")
    pd.DataFrame({"test": best[3]}).to_parquet(OUT / f"{name}_tuned_test.parquet")
    print("BEST", name, round(best[0], 4), best[1], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
