# HSI Futures Quantitative Trading System

Three independent strategy lines sharing the same HI1 minute-bar dataset. Fully reproducible.

## Setup

```bash
conda activate your_env                                  # activate your conda environment
pip install -r requirements.txt                           # pandas, numpy, torch, gymnasium
pip install lightgbm scikit-learn joblib scipy            # for LGBM and GARCH-Bollinger
```

## Data Preparation (shared)

Raw data: `data/hi1_20170701_20200609.csv` — minute-bar OHLCV from 2017-07-03 to 2020-06-09.

```bash
python GRPO/scripts/prepare_data.py
```

Splits into original/day/night datasets. Day session: 09:15-16:30. Night session: 17:15-03:00 (next day). Each dataset is split chronologically 3:1:1 → train/validation/test. Output to `data/`.

## Strategy 1: GRPO (Reinforcement Learning)

```bash
# Full training (30 epochs, group=8, steps=2048)
python GRPO/scripts/train_grpo.py

# Quick smoke test
python GRPO/scripts/train_grpo.py --epochs 1 --group-size 2 --episode-steps 256

# Evaluate 5 model-dataset combinations, select best
python GRPO/scripts/evaluate_models.py

# Dynamic risk-managed backtest
python GRPO/scripts/risk_management.py
python GRPO/scripts/risk_management.py --risk-per-trade 0.003 --max-contracts 3
```

MLP policy network (input → 64 → 64 → 3), Group Relative Policy Optimization. Models → `GRPO/models/`, results → `GRPO/outputs/`.

## Strategy 2: LGBM (Supervised Learning)

```bash
cd LGBMmodel
python model_training.py      # Feature engineering + 5-fold TimeSeriesSplit CV + train + evaluate
python risk_management.py     # Risk-managed backtest
```

LightGBM regressor predicting next-minute return, ~40+ features, threshold-based long/short/flat signals. Model → `LGBMmodel/best_model.pkl`.

## Strategy 3: GARCH-Bollinger (Traditional Quant)

```bash
python submission_garch_bollinger/code/garch_bollinger_first4_from_processed_data.py
```

GARCH(1,1) conditional volatility + Bollinger Bands + volume confirmation. 5 variants (standard/garch BB × contrarian/momentum × with/without volume filter). Grid search over bar-minutes, MA windows, k-values. Results → `outputs/garch_bollinger_first4/`.

## Report

```bash
python report/generate_report_figures.py   # Generate figures
cd report && pdflatex final_report.tex     # Compile LaTeX
```

## Key Configuration

- Initial capital: 5,000,000 (`GRPO/src/config.py` → `MarketConfig.initial_capital`)
- Transaction cost: 10 HKD/contract + 1-point slippage × 50 multiplier = 60 HKD/side
- Day/night session boundaries: `GRPO/src/data.py` → `split_day_night()`
- Risk parameters: `GRPO/src/config.py` → `RiskConfig`, `LGBMmodel/risk_management.py` top section
