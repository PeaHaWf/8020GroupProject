# 模型训练与评估说明

## 训练目标

本项目对三个数据集分别训练强化学习交易策略：

- `original_model`：使用 `data/original_train.csv`
- `day_model`：使用 `data/day_train.csv`
- `night_model`：使用 `data/night_train.csv`

训练脚本为 `GRPO/scripts/train_grpo.py`，核心实现位于 `GRPO/src/grpo.py`。

## 环境设定

交易环境位于 `GRPO/src/trading_env.py`。

- 动作空间：`0=short`、`1=flat`、`2=long`
- 每次只交易 1 张合约
- 使用下一根 bar 的价格变化计算持仓收益
- 每次换仓扣除 transaction cost 和 slippage
- 特征标准化参数由训练集计算并随模型保存，验证、测试和风控复用同一组参数

默认参数在 `GRPO/src/config.py` 的 `MarketConfig` 中修改：

```python
contract_multiplier = 50.0
transaction_cost_per_contract = 10.0
slippage_points = 1.0
initial_capital = 5_000_000.0
```

## GRPO 风格优化

每个 epoch 采样一组轨迹，使用组内收益均值和标准差计算相对优势：

```text
advantage_i = (return_i - mean(group_returns)) / (std(group_returns) + epsilon)
```

策略损失：

```text
loss = -mean(advantage_i * sum(log_prob(actions_i))) - entropy_coef * entropy
```

这保留了 GRPO 的组内相对奖励思想，并用于金融交易环境中的策略梯度优化。

## 评价指标

Max drawdown：

```text
MDD = min(equity_t / max(equity_0...equity_t) - 1)
```

Sharpe ratio：

```text
Sharpe = mean(pnl) / std(pnl) * sqrt(bars_per_year)
```

Log return：

```text
LogReturn = log(final_equity / initial_equity)
```

指标实现在 `GRPO/src/metrics.py`。

## 复现命令

```bash
python GRPO/scripts/prepare_data.py
python GRPO/scripts/train_grpo.py
python GRPO/scripts/evaluate_models.py
python GRPO/scripts/risk_management.py
```

如需快速测试训练流程：

```bash
python GRPO/scripts/train_grpo.py --epochs 1 --group-size 2 --episode-steps 256
```

## 评估组合

`GRPO/scripts/evaluate_models.py` 会评估以下五组：

- `original_model` on `original_test`
- `original_model` on `day_test`
- `original_model` on `night_test`
- `day_model` on `day_test`
- `night_model` on `night_test`

输出文件：

- `outputs/evaluation_results.csv`
- `outputs/best_model.json`
- `outputs/trace_*_test.csv`
