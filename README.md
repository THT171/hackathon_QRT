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
Submission: `ID,Close return` for all 219,157 IDs (train 0–144,224 + test 144,225–219,156).

## Run
```bash
python -m src.run --model benchmark   # reproduce the 0.523 benchmark
python -m src.run --model lgbm        # intraday features + LightGBM (binary, weighted by |y|)
```
