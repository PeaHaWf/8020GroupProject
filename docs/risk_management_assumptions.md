# Risk Management Assumptions

## Initial Capital

Default initial capital:

```text
5,000,000
```

Defined in `GRPO/src/config.py` → `MarketConfig.initial_capital`:

```python
initial_capital: float = 5_000_000.0
```

Changing this value updates it across training, evaluation, and risk management scripts.

## Dynamic Risk Parameters

The risk management layer uses balanced dynamic position sizing. The model provides directional signals only; the risk layer decides whether to trade and how many contracts.

Parameters in `GRPO/src/config.py` → `RiskConfig`:

```python
risk_per_trade = 0.005
max_contracts = 5
protective_drawdown = 0.08
kill_switch_drawdown = 0.12
stop_loss_risk_multiple = 1.0
take_profit_risk_multiple = 1.5
cooldown_bars = 20
volatility_lookback = 60
min_volatility_points = 20.0
```

Definitions:

- `risk_per_trade`: maximum 0.5% of current equity at risk per trade.
- `max_contracts`: maximum position size, default 5 contracts.
- `protective_drawdown`: when drawdown from peak exceeds 8%, reduce to max 1 contract.
- `kill_switch_drawdown`: when drawdown exceeds 12%, stop opening new positions (close-only mode).
- `stop_loss_risk_multiple`: force stop-loss when trade loss reaches 1× the risk budget.
- `take_profit_risk_multiple`: take profit when trade gain reaches 1.5× the risk budget.
- `cooldown_bars`: pause trading for this many bars after a stop-loss.
- `volatility_lookback`: lookback window for recent price volatility estimation.
- `min_volatility_points`: minimum volatility floor to prevent oversized positions in low-volatility regimes.

## Risk Management Outputs

Run:

```bash
python GRPO/scripts/risk_management.py
```

Reads the selected model from `GRPO/outputs/best_model.json` and runs on the corresponding test set. Outputs:

- `GRPO/outputs/risk_equity_curve.csv`: per-bar decisions, P&L, and equity balance
- `GRPO/outputs/risk_trade_ledger.csv`: per-trade entry/exit, P&L, post-trade balance
- `GRPO/outputs/risk_summary.json`: risk management summary metrics
- `GRPO/outputs/dynamic_risk_equity_curve.csv`: dynamic risk per-bar equity curve
- `GRPO/outputs/dynamic_risk_trade_ledger.csv`: dynamic risk per-trade ledger
- `GRPO/outputs/dynamic_risk_summary.json`: dynamic risk summary
- `GRPO/outputs/risk_comparison.csv`: fixed 1-contract baseline vs. dynamic risk comparison

Override risk parameters from the command line:

```bash
python GRPO/scripts/risk_management.py --risk-per-trade 0.003 --max-contracts 3
```

## P&L Formulas

Per-step gross P&L:

```text
gross_pnl = position * (next_price - current_price) * contract_multiplier
```

Transaction cost:

```text
cost = abs(new_position - old_position) * (transaction_cost_per_contract + slippage_points * contract_multiplier)
```

Per-step net P&L:

```text
net_pnl = gross_pnl - cost
```

Equity balance:

```text
equity_t = equity_{t-1} + net_pnl_t
```

With dynamic position sizing, `position` expands to signed contract count:

```text
contracts = signal * floor((equity * risk_per_trade) / volatility_risk_per_contract)
contracts = clip(contracts, -max_contracts, max_contracts)
```

Where:

```text
volatility_risk_per_contract = max(recent_volatility_points, min_volatility_points) * contract_multiplier
```
