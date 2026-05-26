# 风险管理系统假设

## 初始资金

本项目默认初始资金为：

```text
5,000,000
```

该假设定义在 `src/config.py` 的 `MarketConfig.initial_capital`：

```python
initial_capital: float = 5_000_000.0
```

如需修改初始资金，只需要修改这一处。训练、评估和风险管理脚本都会读取同一个配置。

## 风险管理输出

运行：

```bash
python scripts/risk_management.py
```

会读取 `outputs/best_model.json` 中选择出的实测模型，并在对应测试集上输出：

- `outputs/risk_equity_curve.csv`：逐 bar 决策、盈亏和资金余额
- `outputs/risk_trade_ledger.csv`：每笔交易的进出场、盈亏、交易后资金余额
- `outputs/risk_summary.json`：风险管理汇总指标

## 交易盈亏计算

单步毛收益：

```text
gross_pnl = position * (next_price - current_price) * contract_multiplier
```

交易成本：

```text
cost = abs(new_position - old_position) * (transaction_cost_per_contract + slippage_points * contract_multiplier)
```

单步净收益：

```text
net_pnl = gross_pnl - cost
```

资金余额：

```text
equity_t = equity_{t-1} + net_pnl_t
```
