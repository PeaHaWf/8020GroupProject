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

## 动态风险管理参数

新版风险管理默认使用均衡型动态仓位。模型只给出方向信号，风控层决定是否交易以及交易几张合约。

参数定义在 `src/config.py` 的 `RiskConfig`：

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

含义：

- `risk_per_trade`：单笔交易最多承担当前权益的 0.5% 风险。
- `max_contracts`：最大持仓张数，默认最多 5 张。
- `protective_drawdown`：当账户从历史高点回撤超过 8%，进入保护模式，最多 1 张。
- `kill_switch_drawdown`：当回撤超过 12%，停止新开仓，只允许平仓。
- `stop_loss_risk_multiple`：单笔亏损达到风险预算 1 倍时强制止损。
- `take_profit_risk_multiple`：单笔浮盈达到风险预算 1.5 倍时止盈。
- `cooldown_bars`：止损后暂停交易的 bar 数。
- `volatility_lookback`：用于估计近期价格波动的窗口长度。
- `min_volatility_points`：最低波动点数假设，避免低波动时仓位过大。

## 风险管理输出

运行：

```bash
python scripts/risk_management.py
```

会读取 `outputs/best_model.json` 中选择出的实测模型，并在对应测试集上输出：

- `outputs/risk_equity_curve.csv`：逐 bar 决策、盈亏和资金余额
- `outputs/risk_trade_ledger.csv`：每笔交易的进出场、盈亏、交易后资金余额
- `outputs/risk_summary.json`：风险管理汇总指标
- `outputs/dynamic_risk_equity_curve.csv`：新版动态风控逐 bar 资金曲线
- `outputs/dynamic_risk_trade_ledger.csv`：新版动态风控逐笔交易账本
- `outputs/dynamic_risk_summary.json`：新版动态风控汇总
- `outputs/risk_comparison.csv`：固定 1 张旧方案与动态风控新方案对比

命令行可覆盖部分风控参数：

```bash
python scripts/risk_management.py --risk-per-trade 0.003 --max-contracts 3
```

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

新版动态仓位下，`position` 会扩展为带方向的合约张数：

```text
contracts = signal * floor((equity * risk_per_trade) / volatility_risk_per_contract)
contracts = clip(contracts, -max_contracts, max_contracts)
```

其中：

```text
volatility_risk_per_contract = max(recent_volatility_points, min_volatility_points) * contract_multiplier
```
