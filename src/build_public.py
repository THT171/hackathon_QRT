"""Compute public factors for train/test, cache, and run the single-factor IC report on them."""
import pandas as pd

from .data import load
from .public_factors import PUB_FAMILY, build_public, fit_peers
from .research import factor_report

OUT = "data/processed/"

if __name__ == "__main__":
    train, test = load()
    peers = fit_peers(train)
    print("median peer corr:", round(float(peers["C"].stack().median()), 3),
          "mean top-10 corr:", round(float(peers["W"].replace(0, float("nan")).stack().mean()), 3))
    for name, df in (("train", train), ("test", test)):
        f = build_public(df, peers).reset_index(drop=True)
        f.to_parquet(OUT + f"public_{name}.parquet")
        print(name, f.shape)
    pd.Series(PUB_FAMILY).to_csv(OUT + "public_families.csv")
    base = pd.read_parquet(OUT + "factors_train.parquet")
    f = pd.read_parquet(OUT + "public_train.parquet")
    rep, _ = factor_report(f, base.y.to_numpy(), base, base.stock_rv.to_numpy(), PUB_FAMILY)
    rep.to_csv("reports/public_factor_report.csv")
    pd.set_option("display.width", 250)
    print(rep[["family", "IC", "RankIC", "ICIR", "t", "hit", "Q1", "Q5", "LS_bp", "score"]].round(4).to_string())
