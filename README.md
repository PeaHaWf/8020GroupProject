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
python GRPO/scripts/prepare_data.py
python GRPO/scripts/train_grpo.py
python GRPO/scripts/evaluate_models.py
python GRPO/scripts/risk_management.py
```

For a quick smoke run:

```bash
python GRPO/scripts/train_grpo.py --epochs 1 --group-size 2 --episode-steps 256
```

The default training command uses the normal experiment setting in `GRPO/src/config.py`: 30 epochs, group size 8, and 2048 steps per sampled episode.

The risk management script now runs the balanced dynamic position sizing layer by default. You can override common risk parameters:

```bash
python GRPO/scripts/risk_management.py --risk-per-trade 0.003 --max-contracts 3
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
- `outputs/dynamic_risk_equity_curve.csv`
- `outputs/dynamic_risk_trade_ledger.csv`
- `outputs/dynamic_risk_summary.json`
- `outputs/risk_comparison.csv`
