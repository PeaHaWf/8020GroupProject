# Quantitative Trading Strategy Report

## 1. Introduction

This project studies whether intraday Hang Seng Index futures prices contain exploitable short-term patterns after transaction costs and execution frictions. The theoretical starting point is the Efficient Market Hypothesis (EMH). Under weak-form and semi-strong-form efficiency, prices should already reflect historical prices and public information, so persistent abnormal returns should be difficult to obtain. In practice, high-frequency and intraday markets can still show temporary predictability because of liquidity imbalance, volatility clustering, investor overreaction, market microstructure noise, and delayed information diffusion across day and night sessions.

The project therefore compares three different trading approaches. The first is a supervised machine learning strategy based on LightGBM (LGBM), which predicts the next-bar return from engineered price, volatility, volume, and time features. The second is a reinforcement learning strategy based on a GRPO-style policy optimization framework, where a policy network learns whether to be short, flat, or long directly from trading rewards. The third is a traditional and interpretable GARCH-Bollinger strategy, which combines Bollinger Bands with GARCH conditional volatility, volume confirmation, and separate day/night parameter selection.

The purpose is not only to find the best in-sample strategy. The more important objective is to evaluate how model choice, validation design, transaction costs, slippage assumptions, and risk management affect out-of-sample performance. For this reason, the project uses chronological train-validation-test splits and treats the final test period as a genuine out-of-sample evaluation set.

## 2. Trading Algorithms and Rationale

### 2.1 LGBM Supervised Learning Strategy

The LGBM strategy treats trading as a short-horizon return prediction problem. The target variable is the next-minute close-to-close return:

```text
target_t = close_(t+1) / close_t - 1
```

The model is a LightGBM gradient boosting regressor. It is suitable for this task because tree boosting can model nonlinear interactions among technical indicators, volume variables, volatility measures, and time-of-day effects without requiring a strictly linear relationship between features and returns.

The feature set contains 41 variables. They include lagged returns over multiple horizons, log returns, rolling volatility, volume moving averages and volume ratios, candlestick shape variables, moving averages, moving-average deviations, moving-average cross features, RSI-style indicators, and intraday calendar encodings such as minute-of-day and day-of-week sine/cosine transformations. The feature engineering is designed to capture several types of short-term information:

- Momentum and reversal are represented by lagged returns and moving-average deviations.
- Volatility clustering is represented by rolling return volatility.
- Liquidity and participation are represented by volume ratios.
- Intraday seasonality is represented by hour, minute-of-day, and periodic encodings.

The trading rule converts predicted return into position. If the predicted return is larger than the estimated round-trip cost threshold, the strategy goes long. If the predicted return is smaller than the negative cost threshold, it goes short. Otherwise, it stays flat:

```text
if predicted_return_t > total_cost_t:
    position_t = +1
elif predicted_return_t < -total_cost_t:
    position_t = -1
else:
    position_t = 0
```

The threshold includes both transaction cost and slippage. This is important because a model that predicts small positive returns may still be untradable if the expected return does not cover execution costs.

### 2.2 GRPO Reinforcement Learning Strategy

The GRPO strategy formulates intraday trading as a sequential decision problem. The trading environment has three discrete actions:

```text
0 = short
1 = flat
2 = long
```

The environment maps these actions to target positions `{-1, 0, +1}`. Each step uses the current bar's price and the next bar's price to calculate the profit and loss:

```text
gross_pnl_t = position_t * (price_(t+1) - price_t) * contract_multiplier
cost_t = abs(position_t - position_(t-1)) * cost_per_contract
net_pnl_t = gross_pnl_t - cost_t
reward_t = net_pnl_t / initial_capital
```

The observation vector contains standardized market features and the current position. The market features include open, high, low, and close returns, log return, rolling volatility, volume z-score, and intrabar range. Feature mean and standard deviation are computed on the training set and saved with the model, then reused during validation, testing, and risk management. This avoids using future distribution information during evaluation.

The optimization follows a GRPO-style relative reward idea. In each epoch, the training code samples a group of episodes. Each episode produces a total return. The advantage of an episode is its return relative to the group mean and group standard deviation:

```text
advantage_i = (return_i - mean(group_returns)) / (std(group_returns) + epsilon)
```

The policy loss is:

```text
loss = -mean(advantage_i * sum(log_prob(actions_i))) - entropy_coef * entropy
```

This design rewards policies that perform better than other sampled trajectories in the same group. The entropy term encourages exploration and reduces premature convergence to a deterministic policy. Compared with the LGBM strategy, the GRPO model optimizes trading behavior more directly because transaction cost and position changes are already embedded in the environment reward.

### 2.3 GARCH-Bollinger Strategy

The GARCH-Bollinger strategy is included as a traditional benchmark with strong interpretability. A standard Bollinger Band uses a moving average as the center line and a rolling standard deviation as the band width:

```text
ma_t = rolling_mean(close, ma_window).shift(1)
rolling_std_t = rolling_std(close, ma_window).shift(1)
upper_t = ma_t + k * rolling_std_t
lower_t = ma_t - k * rolling_std_t
```

The `.shift(1)` design is important because the band used for the decision at time `t` must be known before observing the trading result at time `t`. This avoids look-ahead bias.

The limitation of standard Bollinger Bands is that rolling standard deviation gives similar weight to observations inside a fixed window. Intraday futures returns often show volatility clustering, so a fixed rolling window can react too slowly when volatility regimes change. The project therefore extends the band width using GARCH(1,1) conditional volatility:

```text
r_t = sigma_t * z_t
sigma_t^2 = omega + alpha * r_(t-1)^2 + beta * sigma_(t-1)^2
```

Since GARCH volatility is return volatility, it must be converted back to price scale before being added to the moving average:

```text
band_width_t = k * close_(t-1) * sigma_t
upper_t = ma_t + band_width_t
lower_t = ma_t - band_width_t
```

Two interpretations of a band breach are tested. The contrarian version treats a breach as overreaction and trades against the move. The momentum version treats a breach as a breakout and trades with the move. Some variants also require volume confirmation:

```text
volume_ok_t = volume_t > vol_ratio * rolling_mean(volume, volume_window).shift(1)
```

The volume rule is only used for new entries. Exits do not require volume confirmation. This reflects the idea that a breakout with stronger participation may be more informative than a price move on weak volume.

## 3. Data Splitting and Experimental Design

The raw data are Hang Seng Index futures one-minute bars from July 2017 to June 2020. The project constructs three datasets:

- `original`: all processed one-minute bars.
- `day`: day-session bars, from 09:15 to 12:00 and 13:00 to 16:30.
- `night`: night-session bars, from 17:15 to the next day 03:00.

All datasets are sorted chronologically and split into train, validation, and test sets using a 3:1:1 ratio. The split is chronological rather than random, which is essential for time-series trading research because random splitting would allow future market regimes to leak into the training process.

The split summary is:

- `original`: 582,100 rows. Train has 349,260 rows, validation has 116,420 rows, and test has 116,420 rows. The full period runs from 2017-07-03 09:14:00 to 2020-06-09 16:29:00.
- `day`: 274,448 rows. Train has 164,668 rows, validation has 54,890 rows, and test has 54,890 rows. The full period runs from 2017-07-03 09:15:00 to 2020-06-09 16:29:00.
- `night`: 306,133 rows. Train has 183,679 rows, validation has 61,227 rows, and test has 61,227 rows. The full period runs from 2017-07-03 17:15:00 to 2020-03-16 23:59:00.

The role of each split is different:

```text
train: fit model parameters
validation: select models, strategies, or trading parameters
test: final out-of-sample evaluation only
```

For LGBM, the training set is used for feature construction, hyperparameter selection with time-series cross-validation, and final model fitting. The validation set is used to select the best model-scenario combination. The test set is used only for final reporting.

For GRPO, each of the original, day, and night models is trained on its corresponding training set and validated during training. The evaluation script then tests five combinations: original model on original test, original model on day test, original model on night test, day model on day test, and night model on night test.

For GARCH-Bollinger, GARCH parameters are fitted from historical data and trading parameters are selected on validation only. For final testing, GARCH is refitted using train plus validation data, and the selected trading parameters are then applied to the test data without using test performance to reselect the strategy family or parameters.

## 4. Parameter Optimization and Hyperparameter Selection

### 4.1 LGBM Hyperparameter Selection

The LGBM script uses `TimeSeriesSplit` with five folds. This preserves time ordering inside cross-validation and is more appropriate than ordinary shuffled K-fold validation for financial data. The search objective is validation RMSE of the next-minute return prediction.

The candidate grid varies the number of trees, maximum depth, number of leaves, learning rate, minimum child samples, subsampling ratio, and column subsampling ratio. The best parameters selected for all three datasets are:

```text
n_estimators = 500
max_depth = 5
num_leaves = 31
learning_rate = 0.02
min_child_samples = 100
subsample = 0.9
colsample_bytree = 0.9
```

These settings are relatively conservative for a high-frequency return prediction task. The shallow depth and minimum child sample requirement help reduce overfitting, while the low learning rate and larger number of estimators allow gradual boosting.

### 4.2 GRPO Training Parameters

The GRPO experiment uses the default training configuration:

```text
epochs = 30
group_size = 8
episode_steps = 2048
learning_rate = 3e-4
entropy_coef = 0.01
max_grad_norm = 1.0
hidden_size = 64
seed = 42
```

The group size controls how many sampled trajectories are compared when calculating relative advantages. The episode length controls the number of bars in each sampled training trajectory. The entropy coefficient prevents the policy from collapsing too quickly into a single action. Gradient clipping is used to stabilize policy gradient training.

These hyperparameters are not tuned over a large grid. They are chosen as a practical baseline for a course project because reinforcement learning training is computationally expensive and noisy. The validation metrics are therefore interpreted as evidence of model behavior rather than as proof of a globally optimal policy.

### 4.3 GARCH-Bollinger Parameter Optimization

The GARCH-Bollinger strategy uses validation-only grid search. The tested parameters include:

- `bar_minutes`: 5, 15, and 20.
- `ma_window`: 20, 40, and 60.
- `k`: 1.5, 2.0, and 2.5.
- `max_hold_bars`: 5, 10, and 20 for momentum variants.
- `volume_window`: 20 and 40 for volume-confirmed variants.
- `vol_ratio`: 0.8, 1.0, and 1.2 for volume-confirmed variants.

The selection rule prioritizes higher validation Sharpe ratio. If two candidates are close, less severe maximum drawdown and lower turnover are preferred. Zero-turnover candidates are not selected unless every candidate in the relevant family does not trade. This avoids choosing a trivial no-trade model.

The key methodological point is that test performance does not feed back into parameter selection. When the test results later show that contrarian variants are more robust than validation-selected momentum variants, that conclusion is treated as an out-of-sample finding rather than a reason to retroactively re-optimize the strategy.

## 5. Backtesting, Slippage, and Risk Management

### 5.1 Backtesting Assumptions

The backtests include both transaction cost and slippage proxies. The common futures assumptions are:

```text
initial_capital = 5,000,000
contract_multiplier = 50 HKD per index point
slippage_points = 1 point
transaction_cost_per_contract = 10 HKD for GRPO
transaction_cost_rate = 0.01% per side for LGBM
```

The exact accounting differs across strategy implementations. GRPO reports PnL in HKD using the futures multiplier and initial capital. LGBM reports return-based metrics and also has a separate risk-management module. GARCH-Bollinger reports ratio returns for the technical strategy tests. Because these frameworks use different accounting conventions, their numerical results should be compared directionally rather than treated as perfectly identical measurement systems.

### 5.2 LGBM Backtesting Results

The LGBM test results are strong across most scenarios. On `model_comparison_test.csv`, the best-performing scenarios are:

- `original_model on original_test`: log return 2.548, total return 1177.73%, Sharpe ratio 24.07, maximum drawdown -35.44%, profit factor 2.95, and 6,584 trades.
- `original_model on night_test`: log return 1.414, total return 311.09%, Sharpe ratio 25.13, maximum drawdown -11.80%, profit factor 4.52, and 2,599 trades.
- `original_model on day_test`: log return 1.124, total return 207.85%, Sharpe ratio 15.29, maximum drawdown -23.35%, profit factor 2.30, and 4,016 trades.
- `day_model on day_test`: log return 1.017, total return 176.40%, Sharpe ratio 7.78, maximum drawdown -35.81%, profit factor 1.72, and 4,964 trades.
- `night_model on night_test`: log return -1.695, total return -81.63%, Sharpe ratio -16.51, maximum drawdown -81.67%, profit factor 0.47, and 7,032 trades.

The most important result is that the original model generalizes better than the specialized night model in this experiment. The night-specific LGBM model performs poorly on the night test set, suggesting either overfitting, regime instability, or insufficient robustness of the night-session training signal.

The LGBM risk-management module uses initial capital of 5,000,000, maximum position ratio of 30%, maximum single-trade loss ratio of 2%, daily drawdown limit of 5%, and a consecutive-loss halt after five losses. Its risk summary reports final capital of 8,571,798.62, total return of 71.44%, maximum drawdown of -4.30%, 2,704 trades, and win rate of 66.72%. This is more conservative than the raw return-style backtest and is a better representation of capital-aware implementation.

### 5.3 GRPO Backtesting Results

The fixed one-contract GRPO evaluation is weak before adding dynamic risk management:

- `original_model on original_test`: total PnL -994,190, final equity 4,005,810, Sharpe ratio -3.78, maximum drawdown -26.46%.
- `original_model on day_test`: total PnL -712,530, final equity 4,287,470, Sharpe ratio -4.59, maximum drawdown -18.37%.
- `original_model on night_test`: total PnL -395,900, final equity 4,604,100, Sharpe ratio -2.47, maximum drawdown -10.36%.
- `day_model on day_test`: total PnL -154,570, final equity 4,845,430, Sharpe ratio -0.83, maximum drawdown -12.63%.
- `night_model on night_test`: total PnL -939,460, final equity 4,060,540, Sharpe ratio -6.18, maximum drawdown -19.18%.

The selected fixed one-contract model is `day_model` on `day_test`, because it has the least negative performance among the tested GRPO combinations. However, the fixed strategy still loses money, which shows that the raw policy signal alone is not sufficient.

The dynamic balanced risk layer materially improves the GRPO result. The comparison is:

- Fixed one-contract method: total PnL -154,570, final equity 4,845,430, log return -0.0314, Sharpe ratio -0.83, maximum drawdown -12.63%, 7,704 trades.
- Dynamic balanced method: total PnL 1,607,190, final equity 6,607,190, log return 0.2786, Sharpe ratio 3.41, maximum drawdown -12.04%, 4,188 trades.

The dynamic layer sizes positions using recent volatility and equity risk budget, reduces maximum contracts in protective drawdown mode, stops new trades under a kill-switch drawdown, applies stop-loss and take-profit rules based on the trade risk budget, and uses cooldown bars after stop-loss events. This result shows that risk management is not a cosmetic add-on; it can change both return and risk characteristics of a model-driven trading strategy.

### 5.4 GARCH-Bollinger Backtesting Results

The GARCH-Bollinger validation results favored momentum strategies with volume confirmation. However, out-of-sample test results were more favorable to contrarian rules. This is an important regime-risk finding.

Representative test results include:

- `standard_bb_contrarian` with original parameters on `original_test`: return 4.88%, Sharpe ratio 1.92, maximum drawdown -2.13%, turnover 122.
- `standard_bb_contrarian` with original parameters on `day_test`: return 1.47%, Sharpe ratio 2.24, maximum drawdown -0.58%, turnover 5.
- `standard_bb_contrarian` with original or night parameters on `night_test`: return 4.02%, Sharpe ratio 1.93, maximum drawdown -2.13%, turnover 126.
- `garch_bb_contrarian_volume` with day parameters on `day_test`: return 3.44%, Sharpe ratio 1.96, maximum drawdown -1.43%, turnover 57.
- `garch_bb_momentum_volume` with day parameters on `day_test`: return 1.64%, Sharpe ratio 0.89, maximum drawdown -1.97%, turnover 125.

The result is not simply that GARCH is always better than standard Bollinger Bands, or that momentum is always worse than contrarian trading. A more accurate interpretation is that the validation period behaved more like a breakout environment, while the test period, especially late 2019 to early 2020, was more favorable to mean-reversion after band breaches. This supports the need for out-of-sample testing and warns against selecting strategy families based only on validation performance.

### 5.5 Slippage Discussion

The project models slippage with a simplified one-point proxy. In real trading, slippage is not constant. It depends on order type, queue position, bid-ask spread, market depth, volatility, news timing, and whether the strategy needs to reverse directly from long to short or short to long.

Slippage can be especially large in the following situations:

- Night sessions, where liquidity may be thinner.
- Market open, close, lunch break restart, and night-session reopen.
- Macro news, overseas market shocks, and high-volatility events.
- Large gaps between bars.
- Direct position flips that require closing the old position and opening the opposite position.
- Market orders or urgent stop orders that consume liquidity.

Therefore, the backtest should be interpreted as a bar-level execution approximation. A live implementation should test different cost assumptions, add spread-sensitive execution logic, and avoid relying on signals whose expected return barely exceeds estimated transaction cost.

### 5.6 Risk Management Features

The project implements risk management at several levels.

For LGBM, the risk manager controls position sizing, maximum exposure, single-trade loss, daily drawdown, and consecutive-loss halt. It adjusts the number of contracts using both capital constraints and volatility estimates.

For GRPO, the dynamic risk module is more explicit. It uses:

- `risk_per_trade = 0.005`, so each trade risks at most 0.5% of equity.
- `max_contracts = 5`, so leverage cannot grow without bound.
- `protective_drawdown = 0.08`, so drawdown beyond 8% reduces maximum contracts to one.
- `kill_switch_drawdown = 0.12`, so drawdown beyond 12% stops new entries.
- `stop_loss_risk_multiple = 1.0`, so a trade is closed after losing one risk budget.
- `take_profit_risk_multiple = 1.5`, so a trade is closed after earning 1.5 risk budgets.
- `cooldown_bars = 20`, so the strategy pauses after a stop-loss event.
- `volatility_lookback = 60` and `min_volatility_points = 20`, so position sizing does not become too aggressive during quiet periods.

For GARCH-Bollinger, risk control is simpler but still present through band width, volume confirmation, maximum holding bars for momentum trades, and transaction-cost-adjusted return calculation. Wider bands and volume filters reduce turnover, while maximum holding periods prevent momentum trades from staying open indefinitely.

## 6. Real-Time Implementation Difficulties

The current project is a backtesting and research framework, not a production trading system. Moving it to real-time trading would introduce several practical problems.

First, real-time data quality is harder than historical CSV backtesting. A live system must detect missing bars, delayed ticks, duplicated bars, lunch breaks, night-session boundaries, exchange holidays, and abnormal timestamps. If the live data feed handles session gaps differently from the backtest, rolling indicators, GARCH volatility, LGBM features, and GRPO observations may become inconsistent.

Second, signals should be generated only from completed bars. Using an unfinished one-minute bar would create a hidden look-ahead problem because the high, low, close, and volume of that bar are not final yet. The live system should finalize the bar, compute features, generate the signal, and execute only afterward.

Third, futures trading has contract lifecycle issues. HSI futures have expiry months, rollover dates, and liquidity migration from the expiring contract to the next active contract. A backtest on a continuous historical series does not fully solve live contract selection. A production system must decide when to roll, how to handle gaps between contracts, and how to avoid trading illiquid contracts close to expiry.

Fourth, margin and forced liquidation are central to futures trading. The backtests assume initial capital and position limits, but a broker will impose initial margin, maintenance margin, intraday margin checks, and forced liquidation rules. A sharp adverse move can trigger margin calls or liquidation before a model's intended stop-loss logic is reached.

Fifth, execution reliability matters. Real orders can be rejected, partially filled, filled at worse prices, or delayed by API latency. A live strategy must track actual broker positions rather than assumed model positions. This is especially important when the strategy flips from long to short, because a failed close order followed by a new open order can create unintended exposure.

Finally, a live system requires monitoring and operational controls. It should include a kill switch, daily loss limit, order throttling, reconciliation between internal and broker positions, logging, alerting, and manual override. These engineering requirements are outside the scope of the backtest but necessary for real capital deployment.

## 7. Additional Discussion

The three strategies have different strengths and weaknesses.

LGBM produces the strongest raw backtest results, especially when the original model is tested across original, day, and night datasets. However, the extremely high Sharpe ratios and returns should be interpreted carefully because intraday machine learning backtests are vulnerable to overfitting, subtle leakage, and optimistic execution assumptions. The poor performance of the night-specific LGBM model also shows that more specialized training does not automatically improve robustness.

GRPO is conceptually attractive because it optimizes trading decisions directly, including position changes and transaction costs. In this experiment, however, the raw fixed one-contract policy is not profitable. Its main contribution is showing how a weak signal can become more usable when combined with a disciplined dynamic risk layer. This suggests that reinforcement learning for trading should be evaluated together with position sizing and risk controls rather than only by raw action accuracy.

GARCH-Bollinger is the most interpretable strategy. It connects directly to volatility clustering and technical trading logic. Its test results are less spectacular than LGBM, but the strategy is easier to explain and diagnose. The difference between validation-favored momentum and test-favored contrarian behavior is also a useful lesson: even a transparent strategy can suffer from regime instability.

Overall, the project supports a cautious view of quantitative strategy development. A strong backtest is not enough. A credible trading strategy must also have a clear economic rationale, clean data splitting, realistic cost assumptions, robust out-of-sample testing, and risk management that can survive adverse market regimes.

## 8. Conclusion

This project compares supervised learning, reinforcement learning, and traditional volatility-based technical trading on intraday HSI futures data. The LGBM strategy achieves the strongest raw test performance, especially for the original model, but its results require careful interpretation because high-frequency prediction is highly sensitive to overfitting and execution assumptions. The GRPO strategy performs poorly under fixed one-contract trading, but the dynamic balanced risk-management layer significantly improves its final equity, Sharpe ratio, and trade count profile. The GARCH-Bollinger strategy is more interpretable and shows that contrarian band-breach rules can be more robust than validation-selected momentum rules in the final test period.

The main lesson is that model complexity alone does not guarantee trading robustness. The final strategy quality depends on the full pipeline: data cleaning, chronological splitting, feature construction, validation discipline, transaction cost modeling, position sizing, drawdown control, and implementation feasibility. For a real trading deployment, the most important next steps would be more conservative execution simulation, walk-forward validation, live paper trading, and broker-level risk-control integration.

## References

- Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383-417.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*.
- Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*.
- Hong Kong Exchanges and Clearing Limited (HKEX). Hang Seng Index Futures contract specifications and trading information.
