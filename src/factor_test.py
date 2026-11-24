"""Single-factor report: python -m src.factor_test"""
import pandas as pd

from .research import factor_report

META = ["ID", "DATE_ID", "STOCK_ID", "COUNTRY", "CLOSE_TIME", "country_code", "summer",
        "half_day", "cs_size", "y"]

if __name__ == "__main__":
    d = pd.read_parquet("data/processed/factors_train.parquet")
    fam = pd.read_csv("data/processed/families.csv", index_col=0).iloc[:, 0].to_dict()
    rep, ric = factor_report(d.drop(columns=META), d.y.to_numpy(), d, d.stock_rv.to_numpy(), fam)
    rep.to_csv("reports/factor_report.csv")
    print(rep.round(4).to_string())
