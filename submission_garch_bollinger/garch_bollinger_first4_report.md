# GARCH-Bollinger Strategy First Four Sections / 前四部分草稿

## 1. Introduction / 引言

This is the traditional and interpretable strategy line for the STAT8020 HSI futures project. 本实验直接使用已经 processed 的 `original/day/night` train-validation-test CSV，不重新切数据，不连接 IBKR，也不使用 capital / final equity / wealth。

The main question is whether GARCH conditional volatility can improve Bollinger Bands. Standard Bollinger Bands use rolling price standard deviation, while GARCH-Bollinger Bands use a conditional volatility forecast. We also compare two interpretations of band breaches: contrarian mean reversion and momentum breakout.

This rule-based strategy is designed as a transparent benchmark against the GRPO / LightGBM lines. 创新点不是单个指标，而是把 Bollinger Bands、GARCH(1,1)、成交量确认、日夜盘参数差异组合成一个可以解释、可以回测、可以比较的系统。

## 2. Detailed Description / 策略描述与逻辑

Standard Bollinger Bands:

```text
ma_t = rolling_mean(close, ma_window).shift(1)
rolling_std_t = rolling_std(close, ma_window).shift(1)
upper_t = ma_t + k * rolling_std_t
lower_t = ma_t - k * rolling_std_t
```

Standard BB is simple and interpretable, but rolling standard deviation may react slowly when volatility clusters. Lec6 GARCH(1,1) models conditional volatility:

```text
r_t = sigma_t * z_t
sigma_t^2 = omega + alpha * r_(t-1)^2 + beta * sigma_(t-1)^2
```

GARCH is estimated by QMLE on percentage returns using multiple optimizer starting points. The fitted return mean is stored and reused when recursively forecasting conditional variance, so the estimation and forecasting scales are consistent. If the `arch` package is unavailable, this standalone SciPy implementation is used directly.

Because GARCH `sigma_t` is return volatility, it cannot be directly added to a price moving average. The price-band formula used here is:

```text
band_width_t = k * close_(t-1) * sigma_t
upper_t = ma_t + band_width_t
lower_t = ma_t - band_width_t
```

All indicators use shifted information, so the signal at bar `t` only uses data available before or at that bar. The executed position follows:

```text
executed_position_t = target_position_(t-1)
```

To avoid session leakage, all rolling moving averages, rolling standard deviations, GARCH recursions, and volume moving averages are computed inside each continuous `segment_id`. A new segment starts at the first row of each file and whenever the timestamp gap is larger than two minutes. The first bar of each segment has zero return and zero executed position, and the last bar of each segment is forced flat.

The rolling windows use `min_periods = ma_window` or `volume_window`, so the strategy does not trade from incomplete Bollinger or volume windows at the start of a segment.

Contrarian version: price above upper band opens short, price below lower band opens long, and the position exits when price crosses back to the moving average. Momentum version: price above upper band opens long, price below lower band opens short, and the position exits after `max_hold_bars`.

Volume confirmation is only used in `_volume` variants:

```text
volume_ma_t = rolling_mean(volume, volume_window).shift(1)
volume_ok_t = volume_t > vol_ratio * volume_ma_t
```

It only gates new entries. Existing positions may exit without volume confirmation. Day and night sessions use separate selected parameters because HSI futures have different liquidity, volatility, and execution noise across trading sessions.

Default parameter grid:

```text
bar_minutes = 5, 15, 20
ma_window = 20, 40, 60
k = 1.5, 2.0, 2.5
max_hold_bars = 5, 10, 20
volume_window = 20, 40
vol_ratio = 0.8, 1.0, 1.2
```

Non-volume variants ignore `volume_window` and `vol_ratio`. Contrarian variants ignore `max_hold_bars` because they exit on a cross back to the moving average.

## 3. In-sample / Out-of-sample Periods

The fixed processed input files are:

```text
data/original_train.csv, data/original_validation.csv, data/original_test.csv
data/day_train.csv,      data/day_validation.csv,      data/day_test.csv
data/night_train.csv,    data/night_validation.csv,    data/night_test.csv
```

Train fits GARCH. Validation selects trading parameters. Test is final out-of-sample evaluation only and is not used for tuning.

Data used:

| dataset | split | rows | start | end |
| --- | --- | --- | --- | --- |
| original | train | 349260 | 2017-07-03 09:14:00 | 2019-04-17 09:58:00 |
| original | validation | 116420 | 2019-04-17 09:59:00 | 2019-10-25 02:38:00 |
| original | test | 116420 | 2019-10-25 02:39:00 | 2020-06-09 16:29:00 |
| day | train | 164668 | 2017-07-03 09:15:00 | 2019-04-16 10:39:00 |
| day | validation | 54890 | 2019-04-16 10:40:00 | 2019-11-20 09:15:00 |
| day | test | 54890 | 2019-11-20 09:16:00 | 2020-06-09 16:29:00 |
| night | train | 183679 | 2017-07-03 17:15:00 | 2019-04-18 00:01:00 |
| night | validation | 61227 | 2019-04-18 00:02:00 | 2019-10-09 20:53:00 |
| night | test | 61227 | 2019-10-09 20:54:00 | 2020-03-16 23:59:00 |

Selected parameters and validation performance:

| variant | data_used_for_params | bar_minutes | ma_window | k | max_hold_bars | volume_window | vol_ratio | validation_sharpe | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_bb_contrarian | original | 15 | 20 | 2.5000 | 0 |  |  | 1.3565 | 119.0000 |
| garch_bb_contrarian | original | 20 | 20 | 1.5000 | 0 |  |  | 0.4092 | 106.0000 |
| garch_bb_momentum | original | 5 | 40 | 1.5000 | 20 |  |  | 0.9394 | 627.0000 |
| garch_bb_contrarian_volume | original | 5 | 60 | 2.5000 | 0 | 40.0000 | 1.2000 | 0.3864 | 223.0000 |
| garch_bb_momentum_volume | original | 15 | 20 | 2.5000 | 5 | 20.0000 | 1.0000 | 1.6353 | 124.0000 |
| standard_bb_contrarian | day | 5 | 20 | 1.5000 | 0 |  |  | 0.4436 | 457.0000 |
| garch_bb_contrarian | day | 5 | 20 | 2.5000 | 0 |  |  | 0.2742 | 326.0000 |
| garch_bb_momentum | day | 5 | 40 | 2.5000 | 5 |  |  | 1.1323 | 44.0000 |
| garch_bb_contrarian_volume | day | 5 | 40 | 2.0000 | 0 | 20.0000 | 1.2000 | 1.3049 | 1.0000 |
| garch_bb_momentum_volume | day | 5 | 20 | 1.5000 | 5 | 40.0000 | 0.8000 | 3.0833 | 14.0000 |
| standard_bb_contrarian | night | 15 | 20 | 2.5000 | 0 |  |  | 1.1259 | 105.0000 |
| garch_bb_contrarian | night | 20 | 20 | 1.5000 | 0 |  |  | 0.2632 | 104.0000 |
| garch_bb_momentum | night | 5 | 60 | 2.0000 | 20 |  |  | 1.4323 | 255.0000 |
| garch_bb_contrarian_volume | night | 5 | 20 | 2.0000 | 0 | 20.0000 | 1.2000 | 0.0738 | 627.0000 |
| garch_bb_momentum_volume | night | 15 | 20 | 2.5000 | 5 | 20.0000 | 1.2000 | 2.2111 | 147.0000 |

## 4. Back-testing Results and Performance Characteristics / 回测结果与表现特征

Backtest formulas:

```text
close_return_t = close_t / close_(t-1) - 1
turnover_t = abs(executed_position_t - executed_position_(t-1))
cost_return_t = turnover_t * cost_points_per_side / close_(t-1)
strategy_return_t = executed_position_t * close_return_t - cost_return_t
test_return = product(1 + strategy_return_t) - 1
log_return = log(1 + test_return)
max_dd = min(cumulative_curve / running_peak - 1)
Sharpe = mean(strategy_return_t) / std(strategy_return_t) * sqrt(periods_per_year)
periods_per_year = average_test_bars_per_day * 252
```

`cost_points_per_side = 0.6`. Sharpe uses dynamic `periods_per_year`, so 5/15/20-minute tests are not annualized with a 1-minute factor.

Test results below are final out-of-sample evaluations only. They are not used to choose the strategy or parameters.

| variant | data_used_for_params | test_dataset | bar_minutes | ma_window | k | test_return | test_sharpe | max_dd | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| garch_bb_contrarian | day | day_test | 5 | 20 | 2.5000 | 0.0203 | 0.4920 | -0.0682 | 407.0000 |
| garch_bb_contrarian | night | night_test | 20 | 20 | 1.5000 | 0.0135 | 0.8087 | -0.0247 | 114.0000 |
| garch_bb_contrarian | original | day_test | 20 | 20 | 1.5000 | -0.0000 | -1.3612 | -0.0000 | 1.0000 |
| garch_bb_contrarian | original | night_test | 20 | 20 | 1.5000 | 0.0077 | 0.4788 | -0.0275 | 99.0000 |
| garch_bb_contrarian | original | original_test | 20 | 20 | 1.5000 | 0.0034 | 0.1861 | -0.0275 | 95.0000 |
| garch_bb_contrarian_volume | day | day_test | 5 | 40 | 2.0000 | 0.0354 | 2.0115 | -0.0143 | 57.0000 |
| garch_bb_contrarian_volume | night | night_test | 5 | 20 | 2.0000 | -0.0141 | -0.2807 | -0.0790 | 676.0000 |
| garch_bb_contrarian_volume | original | day_test | 5 | 60 | 2.5000 | 0.0322 | 2.6095 | -0.0099 | 23.0000 |
| garch_bb_contrarian_volume | original | night_test | 5 | 60 | 2.5000 | 0.0259 | 0.8212 | -0.0446 | 184.0000 |
| garch_bb_contrarian_volume | original | original_test | 5 | 60 | 2.5000 | 0.0514 | 1.2545 | -0.0439 | 187.0000 |
| garch_bb_momentum | day | day_test | 5 | 40 | 2.5000 | 0.0043 | 0.2396 | -0.0380 | 141.0000 |
| garch_bb_momentum | night | night_test | 5 | 60 | 2.0000 | -0.0746 | -1.9372 | -0.1039 | 327.0000 |
| garch_bb_momentum | original | day_test | 5 | 40 | 1.5000 | -0.0018 | -0.0536 | -0.0451 | 155.0000 |
| garch_bb_momentum | original | night_test | 5 | 40 | 1.5000 | -0.0758 | -1.4975 | -0.1202 | 470.0000 |
| garch_bb_momentum | original | original_test | 5 | 40 | 1.5000 | -0.0704 | -1.0571 | -0.1284 | 604.0000 |
| garch_bb_momentum_volume | day | day_test | 5 | 20 | 1.5000 | 0.0185 | 0.9911 | -0.0193 | 125.0000 |
| garch_bb_momentum_volume | night | night_test | 15 | 20 | 2.5000 | 0.0082 | 0.3482 | -0.0319 | 159.0000 |
| garch_bb_momentum_volume | original | day_test | 15 | 20 | 2.5000 | -0.0235 | -2.7594 | -0.0294 | 9.0000 |
| garch_bb_momentum_volume | original | night_test | 15 | 20 | 2.5000 | 0.0064 | 0.2803 | -0.0379 | 114.0000 |
| garch_bb_momentum_volume | original | original_test | 15 | 20 | 2.5000 | -0.0378 | -1.1473 | -0.0708 | 118.0000 |
| standard_bb_contrarian | day | day_test | 5 | 20 | 1.5000 | 0.0330 | 0.7655 | -0.0733 | 524.0000 |
| standard_bb_contrarian | night | night_test | 15 | 20 | 2.5000 | 0.0421 | 2.0166 | -0.0209 | 126.0000 |
| standard_bb_contrarian | original | day_test | 15 | 20 | 2.5000 | 0.0147 | 2.2518 | -0.0058 | 5.0000 |
| standard_bb_contrarian | original | night_test | 15 | 20 | 2.5000 | 0.0421 | 2.0166 | -0.0209 | 126.0000 |
| standard_bb_contrarian | original | original_test | 15 | 20 | 2.5000 | 0.0507 | 1.9890 | -0.0209 | 122.0000 |

Primary model selected by validation Sharpe:

```text
variant = garch_bb_momentum_volume
data_used_for_params = day
validation_return = 0.007717
validation_sharpe = 3.083307
validation_max_dd = -0.000826
turnover = 14.00
```

Interpretation / 结果解释:

The validation-selected primary model is a momentum variant, so validation evidence favors interpreting band breaches as information-driven breakouts.

The test table is then used to report out-of-sample behavior of the validation-selected parameter sets. We do not choose the model by test Sharpe, because that would be data snooping.

Conditional volatility helps adapt band width under volatility clustering in this sample.

Average Sharpe by test dataset: day_test=0.519, night_test=0.306, original_test=0.245. Day session is stronger on average; night may have lower liquidity or higher execution noise.

Transaction cost effect: every position change pays `cost_points_per_side / close_(t-1)` on each side through turnover. Therefore high-turnover band settings can show attractive raw timing but weaker net ratio returns after costs. This is why validation selection uses Sharpe first, then less severe drawdown, then lower turnover.

Short comparison with Alexander backup: Alexander filter is a price-filter trend-following backup and does not explicitly model volatility clustering. GARCH-Bollinger is more interpretable for the course discussion because it directly connects Lec4/5 Bollinger Bands with Lec6 GARCH conditional volatility and session-specific market structure. If Alexander performs better in a table, it can be presented as a simpler robust benchmark; if GARCH-BB performs better, the explanation is that dynamic volatility width added useful adaptation.
