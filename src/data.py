"""Loading the competition files and turning `Price TS` into a price matrix."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
N_BARS = 96  # a full day has 96 prices up to the cutoff
TARGET = "Close return"


def load(raw_dir: Path = RAW):
    x_train = pd.read_parquet(raw_dir / "x_train.parquet")
    x_test = pd.read_parquet(raw_dir / "x_test.parquet")
    y_train = pd.read_csv(raw_dir / "y_train.csv")
    x_train = x_train.merge(y_train, on="ID", how="left")
    return x_train, x_test


def price_matrix(df: pd.DataFrame, col: str = "Price TS") -> np.ndarray:
    """Return an (n_rows, N_BARS) float array, right-aligned so the last column is the cutoff.

    Handles `Price TS` stored either as a list column or as wide columns (`Price TS_0`, ...).
    Short days (early close) are padded with NaN on the left.
    """
    if col in df.columns:
        series = df[col].to_numpy()
        out = np.full((len(df), N_BARS), np.nan)
        for i, p in enumerate(series):
            a = np.asarray(p if p is not None else [], dtype=float)[-N_BARS:]
            if len(a):
                out[i, N_BARS - len(a):] = a
        return out
    wide = sorted([c for c in df.columns if str(c).startswith(col)],
                  key=lambda c: int(str(c).rsplit("_", 1)[-1]))
    a = df[wide].to_numpy(dtype=float)
    out = np.full((len(df), N_BARS), np.nan)
    # right-align each row on its last non-null price
    for i in range(len(a)):
        row = a[i]
        valid = np.where(~np.isnan(row))[0]
        if len(valid):
            row = row[: valid[-1] + 1][-N_BARS:]
            out[i, N_BARS - len(row):] = row
    return out


def ffill_rows(p: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along each row (leading NaNs stay NaN)."""
    valid = ~np.isnan(p)
    idx = np.where(valid, np.arange(p.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = p[np.arange(p.shape[0])[:, None], idx]
    before_first = np.cumsum(valid, axis=1) == 0
    out[before_first] = np.nan
    return out
