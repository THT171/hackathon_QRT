"""Compute factors for train and test once and cache them to data/processed/."""
from pathlib import Path

import pandas as pd

from .data import load
from .factors import build_factors, context, FAMILIES

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    train, test = load()
    # stock-level stats from the training period only (no target used)
    tmp = build_factors(train)
    stock_stats = pd.DataFrame({"rv": tmp["rv"]}).groupby(train["STOCK_ID"].to_numpy()).median()
    meta_cols = ["ID", "DATE_ID", "STOCK_ID", "COUNTRY", "CLOSE_TIME"]
    for name, df in (("train", train), ("test", test)):
        F = build_factors(df, stock_stats)
        C = context(df)
        out = pd.concat([df[meta_cols].reset_index(drop=True), C.reset_index(drop=True),
                         F.reset_index(drop=True)], axis=1)
        if "Close return" in df:
            out["y"] = df["Close return"].to_numpy()
        out.to_parquet(OUT / f"factors_{name}.parquet")
        print(name, out.shape)
    pd.Series(FAMILIES).to_csv(OUT / "families.csv")


if __name__ == "__main__":
    main()
