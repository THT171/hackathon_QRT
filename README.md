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
| + 6 public factors (ridge+lgbm blend) | 0.5447 (ICIR 0.475, Sharpe 14.2) |
| **equal-weight avg of 9 models (zoo, no MLP)** | **0.5451** (ICIR 0.475) |

## Model zoo (`python -m src.zoo <models>`, `python -m src.stack`)
Single models (country-day demeaned): xgb 0.5447, et 0.5439, lgbm 0.5437, logit 0.5435, rf 0.5434,
cat 0.5433, hgb 0.5433, enet 0.5432, ridge 0.5431, mlp 0.5398. OOF correlations 0.86–0.99.
Equal-weight average beats a nested-CV non-negative stacker (0.5440, overfits to RF).

## Hyper-parameter search (`python -m src.tune xgb 24`, `python -m src.tune et 14`)
XGBoost best 0.5456 (lr 0.03, depth 3, min_child_weight 10, subsample 0.7, colsample 0.9,
lambda 50, gamma 1; median trial 0.5441). ExtraTrees flat (best 0.5437, median 0.5434).
Ensemble with tuned XGB: 0.5450, ICIR 0.477 (vs 0.475). Results: reports/tune_*.csv.

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

## Paper-based wide factor set (`python -m src.alpha158`, `python -m src.wide`)
- Qlib Alpha158 (price part, 5-min bars, windows 3/6/12/24/48) + Alpha360-style raw vol-scaled
  return path and 30-min blocks: 153 factors. Harvey-Liu-Zhu (2016) hurdle |t|>3: 99 pass.
  Best new single factor MA3 (RankIC 0.057, t 15.6).
- Gu-Kelly-Xiu (2020) style: ranks of 161 characteristics into ridge / tuned XGB / LightGBM.
  Scores flat (0.5436-0.5439) but fold std halves (XGB 0.0044 -> 0.0022).
- Ensembles: best ICIR = narrow3+wide3 (0.483, score 0.5444); best score still avg9_xgb_tuned (0.5450).
