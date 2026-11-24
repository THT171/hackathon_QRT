"""Model zoo on the factor library, same 4 DATE_ID folds as src/models.py.

    python -m src.zoo xgb cat hgb et rf mlp logit enet

Each model writes data/processed/zoo/<name>.parquet with OOF (train) and fold-averaged test
predictions, so models can be run separately and stacked later (src/stack.py).
Classifiers predict P(y>0) with sample weight |y| (matches the metric); regressors predict y.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metric import weighted_sign_accuracy
from .models import CTX, PUBLIC_ADD, rank_frame
from .research import _corr_by_group, _group_codes, cs_rank

P = "data/processed/"
OUT = Path(P) / "zoo"
OUT.mkdir(exist_ok=True)


def load_xy():
    tr = pd.read_parquet(P + "factors_train.parquet")
    te = pd.read_parquet(P + "factors_test.parquet")
    tr = pd.concat([tr, pd.read_parquet(P + "public_train.parquet")[PUBLIC_ADD]], axis=1)
    te = pd.concat([te, pd.read_parquet(P + "public_test.parquet")[PUBLIC_ADD]], axis=1)
    rep = pd.read_csv("reports/factor_report.csv", index_col=0)
    cand = rep[rep.absICIR > 0.05].index.tolist() + PUBLIC_ADD
    raw = [c for c in cand if c.endswith("_z") or c.startswith("z_")] + \
          ["path_eff", "bars_since_high", "rsj", "beta", "rv", "cn_upper_shadow", "cn_lower_shadow"]
    Rtr, Rte = rank_frame(tr, cand), rank_frame(te, cand)
    # raw features winsorised and standardised with train stats (for MLP / linear)
    lo, hi = tr[raw].quantile(0.01), tr[raw].quantile(0.99)
    Ztr = tr[raw].clip(lo, hi, axis=1)
    Zte = te[raw].clip(lo, hi, axis=1)
    mu, sd = Ztr.mean(), Ztr.std()
    Ztr, Zte = ((Ztr - mu) / sd).fillna(0), ((Zte - mu) / sd).fillna(0)
    Ztr.columns = Zte.columns = [f"z_{c}" for c in raw]
    ctx_tr, ctx_te = tr[CTX].fillna(0), te[CTX].fillna(0)
    return tr, te, Rtr, Rte, Ztr, Zte, ctx_tr, ctx_te


def make(name):
    """Return (feature set, fit_predict(Xa, ya, Xb, Xtest) -> (pred_b, pred_test))."""
    if name == "xgb":
        import xgboost as xgb

        def fp(Xa, ya, Xb, yb, Xt):
            m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=4,
                                  min_child_weight=50, subsample=0.7, colsample_bytree=0.6,
                                  reg_lambda=50, tree_method="hist", n_jobs=2,
                                  early_stopping_rounds=200, eval_metric="logloss")
            wa, wb = np.abs(ya) / np.abs(ya).mean(), np.abs(yb) / np.abs(yb).mean()
            m.fit(Xa, (ya > 0).astype(int), sample_weight=wa,
                  eval_set=[(Xb, (yb > 0).astype(int))], sample_weight_eval_set=[wb],
                  verbose=False)
            return m.predict_proba(Xb)[:, 1] - 0.5, m.predict_proba(Xt)[:, 1] - 0.5
        return "full", fp
    if name == "cat":
        from catboost import CatBoostClassifier

        def fp(Xa, ya, Xb, yb, Xt):
            m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=20,
                                   thread_count=2, verbose=False, od_type="Iter", od_wait=150)
            from catboost import Pool
            wa, wb = np.abs(ya) / np.abs(ya).mean(), np.abs(yb) / np.abs(yb).mean()
            m.fit(Pool(Xa, (ya > 0).astype(int), weight=wa),
                  eval_set=Pool(Xb, (yb > 0).astype(int), weight=wb), use_best_model=True)
            return m.predict_proba(Xb)[:, 1] - 0.5, m.predict_proba(Xt)[:, 1] - 0.5
        return "full", fp
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fp(Xa, ya, Xb, yb, Xt):
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=15,
                                               min_samples_leaf=800, l2_regularization=10,
                                               early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=50, random_state=0)
            m.fit(Xa, (ya > 0).astype(int), sample_weight=np.abs(ya) / np.abs(ya).mean())
            return m.predict_proba(Xb)[:, 1] - 0.5, m.predict_proba(Xt)[:, 1] - 0.5
        return "full", fp
    if name in ("et", "rf"):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        cls = ExtraTreesRegressor if name == "et" else RandomForestRegressor

        def fp(Xa, ya, Xb, yb, Xt):
            m = cls(n_estimators=120, max_depth=10, min_samples_leaf=400, max_features=0.3,
                    max_samples=0.5 if name == "rf" else None, bootstrap=(name == "rf"),
                    n_jobs=2, random_state=0)
            m.fit(Xa, ya)
            return m.predict(Xb), m.predict(Xt)
        return "rank", fp
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        def fp(Xa, ya, Xb, yb, Xt):
            s = ya.std()
            preds_b, preds_t = [], []
            for seed in (0, 1, 2):
                m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                 batch_size=1024, max_iter=60, early_stopping=True,
                                 n_iter_no_change=6, random_state=seed)
                m.fit(Xa, ya / s)
                preds_b.append(m.predict(Xb)); preds_t.append(m.predict(Xt))
            return np.mean(preds_b, 0), np.mean(preds_t, 0)
        return "rank_z", fp
    if name == "logit":
        from sklearn.linear_model import LogisticRegression

        def fp(Xa, ya, Xb, yb, Xt):
            m = LogisticRegression(C=0.01, max_iter=500)
            m.fit(Xa, (ya > 0).astype(int), sample_weight=np.abs(ya) / np.abs(ya).mean())
            return m.decision_function(Xb), m.decision_function(Xt)
        return "rank_z", fp
    if name == "enet":
        from sklearn.linear_model import ElasticNet

        def fp(Xa, ya, Xb, yb, Xt):
            m = ElasticNet(alpha=2e-6, l1_ratio=0.5, max_iter=3000)
            m.fit(Xa, ya)
            return m.predict(Xb), m.predict(Xt)
        return "rank_z", fp
    raise ValueError(name)


def run(name, data):
    tr, te, Rtr, Rte, Ztr, Zte, Ctr, Cte = data
    fs, fp = make(name)
    if fs == "rank":
        Xtr, Xte = Rtr, Rte
    elif fs == "rank_z":
        Xtr, Xte = pd.concat([Rtr, Ztr], axis=1), pd.concat([Rte, Zte], axis=1)
    else:
        Xtr, Xte = pd.concat([Rtr, Ztr, Ctr], axis=1), pd.concat([Rte, Zte, Cte], axis=1)
    Xtr, Xte = Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)
    y = tr["y"].to_numpy()
    codes = _group_codes(tr)
    folds = list(GroupKFold(4).split(tr, y, tr["DATE_ID"]))
    oof, test = np.zeros(len(tr)), np.zeros(len(te))
    t0 = time.time()
    for a, b in folds:
        pb, pt = fp(Xtr[a], y[a], Xtr[b], y[b], Xte)
        oof[b] = pb
        test += pt / len(folds)
    fs_ = [weighted_sign_accuracy(y[b], oof[b]) for _, b in folds]
    ic = _corr_by_group(cs_rank(pd.Series(oof), codes), y, codes)
    ic = ic[np.isfinite(ic)]
    print(f"{name:6s} score {np.mean(fs_):.4f} ±{np.std(fs_):.4f}  RankIC {ic.mean():.4f}  "
          f"ICIR {ic.mean() / ic.std():.3f}  folds {np.round(fs_, 4)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"oof": oof}).to_parquet(OUT / f"{name}_oof.parquet")
    pd.DataFrame({"test": test}).to_parquet(OUT / f"{name}_test.parquet")


if __name__ == "__main__":
    data = load_xy()
    for nm in sys.argv[1:]:
        try:
            run(nm, data)
        except Exception as e:  # keep going with the other models
            print(nm, "FAILED", repr(e), flush=True)
