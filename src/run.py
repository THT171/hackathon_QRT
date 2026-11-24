"""Cross-validate a model and write a submission.

    python -m src.run --model benchmark
    python -m src.run --model lgbm

CV: GroupKFold on DATE_ID (date ids are shuffled, so no time split is possible).
Submission covers train IDs (out-of-fold predictions) and test IDs, as required.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import RAW, TARGET, ffill_rows, load, price_matrix
from .features import build_features
from .metric import weighted_sign_accuracy

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


def benchmark(df):
    p = ffill_rows(price_matrix(df))
    pred = -(p[:, -1] / p[:, -2] - 1)
    return np.nan_to_num(pred)


def no_zero(pred, fallback):
    pred = np.where(pred == 0, fallback, pred)
    return np.where(pred == 0, EPS, pred)


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    verbose=-1, seed=0,
)


def fit_lgbm(Xtr, ytr, Xva=None, yva=None, rounds=2000):
    dtr = lgb.Dataset(Xtr, (ytr > 0).astype(int), weight=np.abs(ytr))
    if Xva is None:
        return lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xva, (yva > 0).astype(int), weight=np.abs(yva))
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["benchmark", "lgbm"], default="benchmark")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args()

    train, test = load(Path(args.raw))
    train = train[train[TARGET].notna()].reset_index(drop=True)
    y = train[TARGET].to_numpy()
    bench_tr, bench_te = benchmark(train), benchmark(test)
    folds = list(GroupKFold(args.folds).split(train, y, train["DATE_ID"]))

    oof = np.zeros(len(train))
    if args.model == "benchmark":
        oof, pred_te = bench_tr, bench_te
    else:
        Xtr, Xte = build_features(train), build_features(test)
        iters = []
        for k, (tr, va) in enumerate(folds):
            m = fit_lgbm(Xtr.iloc[tr], y[tr], Xtr.iloc[va], y[va])
            oof[va] = m.predict(Xtr.iloc[va]) - 0.5
            iters.append(m.best_iteration)
            print(f"fold {k}: {weighted_sign_accuracy(y[va], oof[va]):.4f} (iters {m.best_iteration})")
        m = fit_lgbm(Xtr, y, rounds=int(np.mean(iters) * 1.1))
        pred_te = m.predict(Xte) - 0.5
        imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
        print("top features:\n", imp.head(15).round(0).to_string())

    fold_scores = [weighted_sign_accuracy(y[va], oof[va]) for _, va in folds]
    print(f"{args.model} CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}  {np.round(fold_scores, 4)}")

    oof = no_zero(oof, bench_tr)
    pred_te = no_zero(pred_te, bench_te)
    sub = pd.concat([
        pd.DataFrame({"ID": train["ID"], TARGET: oof}),
        pd.DataFrame({"ID": test["ID"], TARGET: pred_te}),
    ])
    sample_path = Path(args.raw) / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:  # archive ships without it: IDs are train 0..144224 + test 144225..219156
        sample = pd.DataFrame({"ID": np.sort(np.concatenate([train["ID"], test["ID"]]))})
    sub = sample[["ID"]].merge(sub, on="ID", how="left")
    sub[TARGET] = sub[TARGET].fillna(EPS).replace(0, EPS)
    assert len(sub) == len(sample) and sub["ID"].is_unique and np.isfinite(sub[TARGET]).all()
    out = ROOT / "submissions" / f"{args.model}.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, len(sub), "rows")


if __name__ == "__main__":
    main()
