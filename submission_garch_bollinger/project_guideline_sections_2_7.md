## 2. Detailed Description of Trading Algorithm and Rationale / 策略描述与逻辑

This part of the project uses a traditional and interpretable trading strategy instead of a machine learning or reinforcement learning model. The goal is to test whether a course-based volatility model can improve a standard Bollinger Bands strategy on HSI futures intraday data. The strategy combines Bollinger Bands, GARCH(1,1) conditional volatility, volume confirmation, and day/night session-specific parameter selection.

The baseline signal is the standard Bollinger Bands rule. For each bar, the moving average and rolling price standard deviation are calculated only from past data:

```text
ma_t = rolling_mean(close, ma_window).shift(1)
rolling_std_t = rolling_std(close, ma_window).shift(1)

upper_t = ma_t + k * rolling_std_t
lower_t = ma_t - k * rolling_std_t
```

The `.shift(1)` is important because the band used at time `t` must be known before observing the trading decision at time `t`. This avoids look-ahead bias. The rolling calculation is also done within each continuous trading segment, so the indicator does not cross lunch breaks, overnight gaps, or missing-bar gaps.

Standard Bollinger Bands are closely related to the technical trading rules covered in the course. The logic is simple and interpretable, but there is one important limitation: `rolling_std_t` is based on a fixed look-back window and gives similar weight to observations inside that window. When the market has volatility clustering, meaning large movements tend to be followed by large movements and small movements tend to be followed by small movements, this rolling standard deviation may react with delay.

Concretely, when volatility suddenly increases, the rolling window still contains many low-volatility bars from the previous regime. The estimated band width may therefore be too narrow, creating too many false band breaches and excessive turnover. When volatility falls after a high-volatility period, the rolling window may still contain extreme bars, so the band can remain too wide and delay new trading signals.

This affects trading directly: the band may be too narrow when it should be wider, and too wide when it should be narrower.

GARCH(1,1) conditional volatility is used to mitigate this limitation. Unlike rolling standard deviation, GARCH models volatility as a recursive process. Today's conditional variance depends on yesterday's return shock, through the `alpha` term, and yesterday's conditional variance, through the `beta` term. Therefore:

- a large `r_(t-1)` can immediately increase `sigma_t`, so the band becomes wider on the next bar;
- when volatility is persistent, the `beta` term allows high volatility to remain for several periods;
- when shocks fade, the band adjusts through the GARCH recursion instead of waiting mechanically for old extreme bars to leave a fixed rolling window.

In short, rolling standard deviation is a fixed-window historical estimate, while GARCH sigma is a conditional volatility estimate updated from recent shocks and volatility persistence. This is the main motivation for the GARCH-Bollinger strategy: replacing the historical equal-window standard deviation with conditional volatility, so the band width can adapt more directly to volatility clustering. Whether this improves performance is still an empirical question and must be tested out-of-sample.

The main extension is therefore to replace the rolling price standard deviation with GARCH conditional return volatility. The GARCH(1,1) model is:

```text
r_t = sigma_t * z_t
sigma_t^2 = omega + alpha * r_(t-1)^2 + beta * sigma_(t-1)^2
```

Here `sigma_t` is return volatility, not price volatility. Therefore it cannot be directly added to the price moving average. The price-band width is converted back into price scale as:

```text
band_width_t = k * close_(t-1) * sigma_t

upper_t = ma_t + band_width_t
lower_t = ma_t - band_width_t
```

This is the key formula of the GARCH-Bollinger strategy because it keeps the units consistent: the moving average and bands are in price level, while `sigma_t` is a return-volatility forecast.

Two interpretations of band breaches are tested:

```text
Contrarian:
if close_t > upper_t: short
if close_t < lower_t: long
exit when price crosses back to ma_t

Momentum:
if close_t > upper_t: long
if close_t < lower_t: short
exit after max_hold_bars
```

The contrarian version treats a band breach as temporary overreaction. The momentum version treats a band breach as an information-driven breakout. Testing both versions is useful because the same technical event can have different meanings under different market regimes.

Some variants add volume confirmation:

```text
volume_ma_t = rolling_mean(volume, volume_window).shift(1)
volume_ok_t = volume_t > vol_ratio * volume_ma_t
```

Volume confirmation is only used for new entries. Closing an existing position does not require volume confirmation. The logic is that a breakout with higher-than-normal volume may be more reliable than a price move with weak participation.

The strategy is also estimated separately on `original`, `day`, and `night` data. This is because day and night sessions have different liquidity, participant structure, and news exposure. A single parameter setting may be too coarse for both sessions. The day session is usually more liquid and locally information-rich, while the night session can be more affected by overseas market movements and lower liquidity.

The backtest allows long, short, and flat positions:

```text
target_position_t in {-1, 0, +1}
```

Trading returns use the previous bar's target position:

```text
executed_position_t = target_position_(t-1)
```

This means the signal is generated first, and the position is applied only from the next bar. This is another protection against look-ahead bias.

## 3. Specification of In-sample and Out-of-sample Periods / 样本内外划分

The project uses the processed split data prepared by the group framework. No raw CSV is re-split in this strategy script. There are three datasets:

- `original`: all processed HSI futures 1-minute bars
- `day`: day-session bars
- `night`: night-session bars

We do not manually split the sample into bull, bear, and sideways market regimes. The provided dataset covers July 2017 to June 2020, which is useful for intraday backtesting but still limited for adding another subjective regime-classification layer. Visually, the early part of the sample has an upward bias, while the later part is flatter and more volatile. Therefore, instead of defining market regimes by hand, we preserve the chronological order and use a simple 3:1:1 train-validation-test split.

This assumption makes the out-of-sample test stricter. If validation and test are in different market environments, the selected parameters must survive that change. The difference between validation and test performance is therefore treated as useful robustness evidence, not as a reason to re-select parameters using the test set.

Each dataset is already split into `train`, `validation`, and `test`. The split summary used in this strategy is:

| Dataset | Split | Rows | Start | End |
|---|---:|---:|---|---|
| original | train | 349260 | 2017-07-03 09:14:00 | 2019-04-17 09:58:00 |
| original | validation | 116420 | 2019-04-17 09:59:00 | 2019-10-25 02:38:00 |
| original | test | 116420 | 2019-10-25 02:39:00 | 2020-06-09 16:29:00 |
| day | train | 164668 | 2017-07-03 09:15:00 | 2019-04-16 10:39:00 |
| day | validation | 54890 | 2019-04-16 10:40:00 | 2019-11-20 09:15:00 |
| day | test | 54890 | 2019-11-20 09:16:00 | 2020-06-09 16:29:00 |
| night | train | 183679 | 2017-07-03 17:15:00 | 2019-04-18 00:01:00 |
| night | validation | 61227 | 2019-04-18 00:02:00 | 2019-10-09 20:53:00 |
| night | test | 61227 | 2019-10-09 20:54:00 | 2020-03-16 23:59:00 |

The role of each period is:

```text
train: fit GARCH parameters
validation: select trading parameters
test: final out-of-sample evaluation only
```

For final test evaluation, GARCH is re-fitted using `train + validation`, and the selected trading parameters are then applied to the test set. The test set is not used to choose `bar_minutes`, `ma_window`, `k`, `max_hold_bars`, `volume_window`, `vol_ratio`, or the strategy family. This is necessary to avoid data snooping. In other words, a model that looks best on test is not selected after seeing test performance.

## 4. Back-testing Results and Performance Characteristics / 回测结果与表现特征

The backtest uses ratio return only. It does not report initial capital, final equity, or wealth. The key return formula is:

```text
close_return_t = close_t / close_(t-1) - 1
strategy_return_t = executed_position_t * close_return_t - cost_return_t
cumulative_return = product(1 + strategy_return_t) - 1
```

On validation, the strongest selected momentum models are all `garch_bb_momentum_volume`. Their validation Sharpe ratios are:

| Parameter Dataset | Selected Momentum Variant | Validation Sharpe | Validation Return | Max DD | Turnover |
|---|---|---:|---:|---:|---:|
| day | garch_bb_momentum_volume | 3.08 | 0.77% | -0.08% | 14 |
| night | garch_bb_momentum_volume | 2.21 | 3.87% | -1.48% | 147 |
| original | garch_bb_momentum_volume | 1.64 | 3.04% | -1.35% | 124 |

This suggests that during the validation period, band breaches with volume confirmation behaved more like breakouts than temporary mispricing. In plain terms, when price broke the band with stronger volume, following the move worked well on validation.

However, the directional out-of-sample tests show a different pattern. When the best validation momentum and contrarian settings are tested on their corresponding test sets, contrarian rules are more stable:

| Params Source | Test Dataset | Momentum Variant | Momentum Sharpe | Momentum Return | Contrarian Variant | Contrarian Sharpe | Contrarian Return |
|---|---|---|---:|---:|---|---:|---:|
| original | original_test | garch_bb_momentum_volume | -1.15 | -3.78% | standard_bb_contrarian | 1.99 | 5.07% |
| original | day_test | garch_bb_momentum_volume | -2.76 | -2.35% | standard_bb_contrarian | 2.25 | 1.47% |
| original | night_test | garch_bb_momentum_volume | 0.28 | 0.64% | standard_bb_contrarian | 2.02 | 4.21% |
| day | day_test | garch_bb_momentum_volume | 0.99 | 1.85% | garch_bb_contrarian_volume | 2.01 | 3.54% |
| night | night_test | garch_bb_momentum_volume | 0.35 | 0.82% | standard_bb_contrarian | 2.02 | 4.21% |

The main result is therefore not simply "momentum is best" or "contrarian is best". The more precise conclusion is:

```text
Validation favored momentum + volume, but test performance favored contrarian rules.
```

One possible market-regime explanation is that the validation period behaved more like a news-driven breakout environment, while the test period behaved more like a high-volatility overreaction environment. The test period includes late 2019 to early 2020, when market stress increased sharply. In such an environment, band breaches may reflect temporary panic or forced trading rather than clean trend continuation. That can make mean reversion more effective than momentum.

This is an interpretation, not causal proof. The backtest does not identify the exact news event or order-flow reason behind each trade. It only shows that the strategy's best validation behavior did not fully carry over to the test period. This is useful because it highlights parameter and regime risk.

Across datasets, day and night sessions behave differently. Day-session validation momentum is the strongest by Sharpe, but test results show that night-session contrarian can still be robust. This supports the decision to test session-specific parameters instead of assuming the same rule works equally well in all trading hours.

## 5. Optimization of Trading Parameters / 参数优化

Parameter optimization is done on validation only. The tested parameter grid is deliberately limited to reduce overfitting:

| Parameter | Values | Role |
|---|---|---|
| `bar_minutes` | 5, 15, 20 | Resampled bar frequency |
| `ma_window` | 20, 40, 60 | Moving-average window for the band center |
| `k` | 1.5, 2.0, 2.5 | Band-width multiplier |
| `max_hold_bars` | 5, 10, 20 | Exit rule for momentum variants |
| `volume_window` | 20, 40 | Volume moving-average window |
| `vol_ratio` | 0.8, 1.0, 1.2 | Volume confirmation threshold |

The parameter `k` controls how wide the Bollinger band is. A smaller `k` gives narrower bands and more frequent signals. A larger `k` gives wider bands and fewer but more extreme signals.

The grid is applied differently across strategy families:

- Standard Bollinger contrarian uses `bar_minutes`, `ma_window`, and `k`.
- GARCH-Bollinger contrarian uses `bar_minutes`, `ma_window`, and `k`.
- GARCH-Bollinger momentum also uses `max_hold_bars`.
- Volume variants additionally use `volume_window` and `vol_ratio`.

The selection rule is:

```text
primary criterion: higher validation Sharpe
tie-breaker 1: less severe max drawdown
tie-breaker 2: lower turnover
```

Parameter settings with zero turnover are not selected unless all candidates for that strategy family do not trade. This prevents the optimization from choosing a trivial no-trade result.

For the directional comparison, the selection is done by family:

```text
momentum:
best of garch_bb_momentum and garch_bb_momentum_volume on validation

contrarian:
best of standard_bb_contrarian, garch_bb_contrarian, and garch_bb_contrarian_volume on validation
```

The selected parameters are then tested on the matching out-of-sample set:

```text
original momentum / contrarian -> original_test, day_test, night_test
day momentum / contrarian -> day_test
night momentum / contrarian -> night_test
```

The important point is that test performance does not feed back into the parameter selection step. If the final discussion says that contrarian worked better on test, that is an out-of-sample finding, not a reason to retrospectively choose contrarian using test data.

## 6. Comments on Slippage in Real Trading / 实盘滑点讨论

The backtest includes transaction cost through a points-per-side proxy. The formula is:

```text
turnover_t = abs(executed_position_t - executed_position_(t-1))
cost_return_t = turnover_t * cost_points_per_side / close_(t-1)
strategy_return_t = executed_position_t * close_return_t - cost_return_t
```

The default setting is:

```text
cost_points_per_side = 0.6
```

This cost is deducted whenever the executed position changes. For example, moving from flat to long has turnover 1, and moving from long to short has turnover 2 because it closes the long and opens the short.

This is not a full order-book slippage model. The backtest does not simulate bid-ask queue priority, partial fills, market order impact, or tick-by-tick execution. The transaction cost is used as a conservative proxy for commission, spread, and normal execution friction.

In real trading, slippage can be larger than this proxy, especially in these situations:

- During night sessions, when liquidity may be thinner.
- Around macro news or overseas market shocks.
- Near session open or close.
- When a price gap occurs between bars.
- When the strategy flips directly from long to short or short to long.
- When a market order is used instead of a carefully placed limit order.

Because the strategy uses intraday bars, execution quality matters. A signal that looks profitable at close-to-close bar prices may be weaker after realistic execution delay and spread. This is especially relevant for high-turnover settings such as narrow bands or short bar intervals.

## 7. Implementation Difficulty and Real-time Trading Issues / 实盘实现难点

The current implementation is designed for backtesting, not live trading. It deliberately avoids IBKR connection and real-time order placement. Turning it into a live system would introduce several additional problems.

First, real-time data handling is harder than processed historical CSV backtesting. The live system must detect missing bars, delayed ticks, lunch breaks, night-session boundaries, and exchange holidays. The current backtest protects against this by creating new segments when the timestamp gap is larger than two minutes. A live version would need the same logic online.

Second, indicators must be updated only after a bar is complete. In live trading, using an unfinished bar can create a hidden look-ahead problem because the bar's high, low, close, and volume are not final yet. The live system should generate signals only from completed bars and apply the position from the next bar.

Third, GARCH updating must be done recursively and consistently. The model can be fitted offline using historical train data, but live conditional volatility must be updated one bar at a time:

```text
sigma_t^2 = omega + alpha * r_(t-1)^2 + beta * sigma_(t-1)^2
```

If the return scale, missing-bar handling, or session reset logic differs between backtest and live trading, the live band width may not match the tested strategy.

Fourth, futures execution introduces practical constraints. The strategy can be long, short, or flat in the backtest, but real HSI futures trading requires margin, contract selection, rollover handling, and risk checks. Short selling is mechanically allowed in futures, but it still requires margin control and position limits.

Fifth, latency and order execution can change the result. The backtest assumes the next bar execution is feasible after a signal is formed. A real broker connection may suffer delay, rejected orders, partial fills, or execution at a worse price.

Finally, a real system would need risk management beyond this report section, such as maximum position size, daily stop loss, kill switch, order rejection handling, and monitoring. For this reason, the current project keeps the GARCH-Bollinger strategy as a historical backtest and comparison framework. It is suitable for evaluating the traditional strategy idea, but not sufficient as a production live trading system.
