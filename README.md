# QRT x ETH Hackathon (24 Nov 2026)

Kaggle: https://www.kaggle.com/competitions/qrt-x-eth-hackathon-24-11-2026

## Layout
- `data/raw/` – competition files (not committed)
- `data/processed/` – derived features
- `src/` – feature and model code
- `notebooks/` – exploration
- `submissions/` – generated submission CSVs (not committed)

## Setup
```bash
pip install -r requirements.txt
kaggle competitions download -c qrt-x-eth-hackathon-24-11-2026 -p data/raw
```

## Task
386 European stocks (DEU, ESP, FRA, GBR, ITA, NLD), synthetic 5-min prices from the open to the
cutoff (30 min before close, up to 96 prices). Predict the cutoff→close return **relative to the
country-day mean**. Score = |y|-weighted accuracy of the predicted sign (0 is always wrong).
Benchmark: `-(last 5-min return)`, CV 0.523 (4 folds by DATE_ID).

## Results (4-fold CV by DATE_ID)
| model | CV |
|---|---|
| benchmark | 0.5227 ± 0.0018 |
| lgbm v1 (intraday features) | 0.5414 ± 0.0022 |
| composite (ICIR-weighted, 10 factors) | 0.5396 |
| ridge on factor ranks | 0.5432 |
| lgbm v2 on factor library | 0.5437 |
| ridge + lgbm blend, country-day demeaned | 0.5446 |
| **+ 6 public factors (blend)** | **0.5447** (ICIR 0.475, Sharpe 14.2) |

## Public factors (src/public_factors.py, `python -m src.build_public`)
Adapted from: Gao-Han-Li-Zhou (2018 JFE) market intraday momentum; Baltussen-Da-Lammers-Martens
(2025) end-of-day reversal; Kakushadze (2016) 101 Formulaic Alphas; technical indicators
(RSI, Bollinger, MACD, stochastic); Chinese broker HF factors (ideal reversal, ideal amplitude,
shadows, surge moments); connected-stock peers (Cohen-Frazzini 2008, Hou 2007).
Findings: EOD reversal has the *opposite* sign here (continuation); Gao market momentum ~0.
Kept by incremental IC (partial |t|>2 vs current model, reports/public_incremental.csv):
a101_9_signcons, cn_upper_shadow, cn_lower_shadow, ta_boll_b, peer_res6_gap, ta_stoch_k.

## Research pipeline
```bash
python -m src.build_cache   # factor library -> data/processed/factors_*.parquet
python -m src.factor_test  # single-factor IC report
python -m src.build_public  # public factors + IC + incremental test
python -m src.models        # composite / ridge / lgbm / ensemble CV + test preds
python -m src.backtest      # country-neutral long-short backtest on OOF preds
python -m src.submit        # final submission
```
Reports in `reports/`: factor_report.csv (IC, RankIC, ICIR, t, quintiles), factor_corr.csv,
model_cv.csv, backtest.csv, composite_weights.csv, ridge_coefs.csv, lgbm_importance.csv.
Submission: `ID,Close return` for all 219,157 IDs (train 0–144,224 + test 144,225–219,156).

## Run
```bash
python -m src.run --model benchmark   # reproduce the 0.523 benchmark
python -m src.run --model lgbm        # intraday features + LightGBM (binary, weighted by |y|)
```
