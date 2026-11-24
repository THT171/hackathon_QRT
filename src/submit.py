"""Write the final submission: ridge + LightGBM z-blend, demeaned within country-day.
Train IDs get out-of-fold predictions, test IDs the full-refit predictions."""
import numpy as np
import pandas as pd

from .research import _group_codes

P = "data/processed/"


def blend(df, ref):
    z = lambda c: (df[c] - ref[c].mean()) / ref[c].std()  # scale with train OOF stats
    b = z("ridge") + z("lgbm")
    return b - b.groupby(_group_codes(df)).transform("mean")


def main():
    tr, te = pd.read_parquet(P + "preds_train.parquet"), pd.read_parquet(P + "preds_test.parquet")
    sub = pd.concat([pd.DataFrame({"ID": tr.ID, "Close return": blend(tr, tr)}),
                     pd.DataFrame({"ID": te.ID, "Close return": blend(te, tr)})]).sort_values("ID")
    sub["Close return"] = sub["Close return"].replace(0, 1e-9)
    assert len(sub) == 219157 and sub.ID.is_unique and np.isfinite(sub["Close return"]).all()
    assert (sub.ID.to_numpy() == np.arange(219157)).all()
    sub.to_csv("submissions/blend_ridge_lgbm.csv", index=False)
    print(sub.describe().round(4), "\npositive share test:", (sub[sub.ID >= 144225]["Close return"] > 0).mean())


if __name__ == "__main__":
    main()
