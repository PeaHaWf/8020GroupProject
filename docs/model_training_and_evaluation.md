# Model Training and Evaluation

## Training Objectives

Three RL trading policies are trained, one per dataset:

- `original_model`: trained on `data/original_train.csv`
- `day_model`: trained on `data/day_train.csv`
- `night_model`: trained on `data/night_train.csv`

Training script: `GRPO/scripts/train_grpo.py`. Core implementation: `GRPO/src/grpo.py`.

## Environment

Trading environment: `GRPO/src/trading_env.py`.

- Action space: `0=short`, `1=flat`, `2=long`
- Single contract per trade
- P&L based on next bar's price change
- Transaction cost and slippage deducted on every position change
- Feature normalization parameters computed from training set, saved with the model, and reused across validation, test, and risk management

Default parameters in `GRPO/src/config.py` → `MarketConfig`:

```python
contract_multiplier = 50.0
transaction_cost_per_contract = 10.0
slippage_points = 1.0
initial_capital = 5_000_000.0
```

## GRPO Optimization

Each epoch samples a group of trajectories. Relative advantages are computed from within-group return mean and std:

```text
advantage_i = (return_i - mean(group_returns)) / (std(group_returns) + epsilon)
```

Policy loss:

```text
loss = -mean(advantage_i * sum(log_prob(actions_i))) - entropy_coef * entropy
```

This preserves GRPO's within-group relative reward idea, applied to policy gradient optimization in a financial trading environment.

## Evaluation Metrics

Max drawdown:

```text
MDD = min(equity_t / max(equity_0...equity_t) - 1)
```

Sharpe ratio:

```text
Sharpe = mean(pnl) / std(pnl) * sqrt(bars_per_year)
```

Log return:

```text
LogReturn = log(final_equity / initial_equity)
```

Metrics implementation: `GRPO/src/metrics.py`.

## Reproduction Commands

```bash
python GRPO/scripts/prepare_data.py
python GRPO/scripts/train_grpo.py
python GRPO/scripts/evaluate_models.py
python GRPO/scripts/risk_management.py
```

Quick smoke test:

```bash
python GRPO/scripts/train_grpo.py --epochs 1 --group-size 2 --episode-steps 256
```

## Evaluation Combinations

`GRPO/scripts/evaluate_models.py` evaluates the following five combinations:

- `original_model` on `original_test`
- `original_model` on `day_test`
- `original_model` on `night_test`
- `day_model` on `day_test`
- `night_model` on `night_test`

Output files:

- `GRPO/outputs/evaluation_results.csv`
- `GRPO/outputs/best_model.json`
- `GRPO/outputs/trace_*_test.csv`
