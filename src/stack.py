"""Combine the model zoo: correlations, simple blends, and a non-negative stacker.

The stacker weights are evaluated with nested CV: for each of the 4 DATE_ID folds, weights are
fit on the other 3 folds' OOF predictions and scored on the held-out fold.
Writes submissions/stack_<best>.csv and reports/stack_results.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from .metric import weighted_sign_accuracy
from .research import _corr_by_group, _group_codes, cs_rank

P = "data/processed/"
ZOO = ["ridge", "lgbm", "enet", "logit", "xgb", "cat", "hgb", "et", "rf", "mlp"]


def load():
    tr = pd.read_parquet(P + "preds_train.parquet")
    te = pd.read_parquet(P + "preds_test.parquet")
    for m in ZOO[2:]:
        tr[m] = pd.read_parquet(f"{P}zoo/{m}_oof.parquet")["oof"].to_numpy()
        te[m] = pd.read_parquet(f"{P}zoo/{m}_test.parquet")["test"].to_numpy()
    return tr, te


def zs(tr, te, models):
    """z-score each model with train OOF stats, then demean within country-day."""
    ct, ce = _group_codes(tr), _group_codes(te)
    Ztr, Zte = {}, {}
    for m in models:
        mu, sd = tr[m].mean(), tr[m].std()
        a, b = (tr[m] - mu) / sd, (te[m] - mu) / sd
        Ztr[m] = (a - a.groupby(ct).transform("mean")).to_numpy()
        Zte[m] = (b - b.groupby(ce).transform("mean")).to_numpy()
    return pd.DataFrame(Ztr), pd.DataFrame(Zte)


def evaluate(p, y, codes, folds):
    fs = [weighted_sign_accuracy(y[b], p[b]) for _, b in folds]
    ic = _corr_by_group(cs_rank(pd.Series(p), codes), y, codes)
    ic = ic[np.isfinite(ic)]
    return dict(score=np.mean(fs), score_std=np.std(fs), RankIC=ic.mean(), ICIR=ic.mean() / ic.std())


def main():
    tr, te = load()
    y = tr["y"].to_numpy()
    codes = _group_codes(tr)
    folds = list(GroupKFold(4).split(tr, y, tr["DATE_ID"]))
    Ztr, Zte = zs(tr, te, ZOO)

    print("OOF prediction correlation:\n", Ztr.corr().round(2).to_string())
    Ztr.corr().to_csv("reports/zoo_corr.csv")

    rows, test_preds = {}, {}
    for m in ZOO:
        rows[m] = evaluate(Ztr[m].to_numpy(), y, codes, folds)
    blends = {
        "avg_all": ZOO,
        "avg_no_mlp": [m for m in ZOO if m != "mlp"],
        "avg_trees": ["lgbm", "xgb", "cat", "hgb", "et", "rf"],
        "avg_linear": ["ridge", "enet", "logit"],
        "avg_top5": ["enet", "xgb", "et", "lgbm", "cat"],
        "linear+trees": ["ridge", "enet", "logit", "lgbm", "xgb", "cat", "et", "rf"],
    }
    for name, ms in blends.items():
        rows[name] = evaluate(Ztr[ms].mean(1).to_numpy(), y, codes, folds)
        test_preds[name] = Zte[ms].mean(1).to_numpy()

    # nested-CV non-negative stacker (target sign*|y| = y, fit by least squares, weights >= 0)
    oof_stack = np.zeros(len(tr))
    for a, b in folds:
        lr = LinearRegression(positive=True, fit_intercept=False).fit(Ztr.iloc[a], y[a])
        oof_stack[b] = lr.predict(Ztr.iloc[b])
    rows["stack_nnls"] = evaluate(oof_stack, y, codes, folds)
    lr = LinearRegression(positive=True, fit_intercept=False).fit(Ztr, y)
    w = pd.Series(lr.coef_, index=ZOO)
    print("stacker weights (normalised):", (w / w.sum()).round(3).to_dict())
    test_preds["stack_nnls"] = lr.predict(Zte)

    res = pd.DataFrame(rows).T.sort_values("score", ascending=False)
    print(res.round(4).to_string())
    res.to_csv("reports/stack_results.csv")

    best = max(test_preds, key=lambda k: res.loc[k, "score"])
    p = np.where(test_preds[best] == 0, 1e-9, test_preds[best])
    oof_best = oof_stack if best == "stack_nnls" else Ztr[blends[best]].mean(1).to_numpy()
    sub = pd.concat([pd.DataFrame({"ID": tr.ID, "Close return": oof_best}),
                     pd.DataFrame({"ID": te.ID, "Close return": p})]).sort_values("ID")
    sub["Close return"] = sub["Close return"].replace(0, 1e-9)
    assert len(sub) == 219157 and sub.ID.is_unique and np.isfinite(sub["Close return"]).all()
    sub.to_csv(f"submissions/stack_{best}.csv", index=False)
    print("best:", best, "-> submissions/stack_%s.csv" % best)


if __name__ == "__main__":
    main()
