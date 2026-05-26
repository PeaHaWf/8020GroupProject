# Quantitative Trading Pipeline

This repository implements the first three parts of `前四布操作.md`:

1. Split raw HI futures data into original/day/night datasets and 3:1:1 train/validation/test splits.
2. Train and evaluate GRPO-style reinforcement learning trading policies.
3. Run a risk management report with an initial capital assumption of 5,000,000.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/prepare_data.py
python scripts/train_grpo.py
python scripts/evaluate_models.py
python scripts/risk_management.py
```

For a quick smoke run:

```bash
python scripts/train_grpo.py --epochs 1 --group-size 2 --episode-steps 256
```

## Main Outputs

- `data/split_summary.csv`
- `models/original_model.pt`
- `models/day_model.pt`
- `models/night_model.pt`
- `outputs/evaluation_results.csv`
- `outputs/best_model.json`
- `outputs/risk_equity_curve.csv`
- `outputs/risk_trade_ledger.csv`
- `outputs/risk_summary.json`
