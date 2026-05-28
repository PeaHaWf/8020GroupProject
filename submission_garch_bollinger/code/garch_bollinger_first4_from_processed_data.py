"""GARCH-Bollinger first-four-section experiment on processed HSI split data.

This entrypoint orchestrates the standalone strategy experiment. It does not
import teammate src/ or scripts/ modules, does not read raw CSV files, and does
not use capital/equity accounting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from garch_bollinger_backtest import (
    evaluate_validation_grid,
    run_core_6_tests,
    run_directional_10_tests,
    run_original_best_on_day_night_tests,
    run_test_combinations,
    select_core_6_params,
    select_directional_family_params,
    select_original_family_params,
    select_params,
)
from garch_bollinger_data import load_processed_split, prepare_bars, write_data_summary
from garch_bollinger_strategy import estimate_garch_models


DATASETS = ("original", "day", "night")
SPLITS = ("train", "validation", "test")
VARIANTS = (
    "standard_bb_contrarian",
    "garch_bb_contrarian",
    "garch_bb_momentum",
    "garch_bb_contrarian_volume",
    "garch_bb_momentum_volume",
)

DEFAULT_BAR_MINUTES = (5, 15, 20)
DEFAULT_MA_WINDOWS = (20, 40, 60)
DEFAULT_K_VALUES = (1.5, 2.0, 2.5)
DEFAULT_MAX_HOLD_BARS = (5, 10, 20)
DEFAULT_VOLUME_WINDOWS = (20, 40)
DEFAULT_VOL_RATIOS = (0.8, 1.0, 1.2)
DEFAULT_COST_POINTS_PER_SIDE = 0.6


def markdown_table(df: pd.DataFrame, float_digits: int = 4, max_rows: int | None = None) -> str:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)

def summarize_validation_choice(selected_params: pd.DataFrame) -> tuple[pd.Series, str]:
    best = selected_params.sort_values(
        ["validation_sharpe", "validation_return", "max_dd", "turnover"],
        ascending=[False, False, False, True],
    ).iloc[0]
    variant = str(best["variant"])
    direction_text = (
        "The validation-selected primary model is a momentum variant, so validation evidence favors interpreting band breaches as information-driven breakouts."
        if "momentum" in variant
        else "The validation-selected primary model is a contrarian variant, so validation evidence favors interpreting band breaches as temporary overreaction."
    )
    return best, direction_text

def comparison_text(test_results: pd.DataFrame) -> str:
    garch_rows = test_results.loc[test_results["variant"].str.startswith("garch_")]
    standard_rows = test_results.loc[test_results["variant"].eq("standard_bb_contrarian")]
    garch_best = garch_rows["test_sharpe"].max()
    standard_best = standard_rows["test_sharpe"].max()
    if garch_best > standard_best:
        vol_text = "Conditional volatility helps adapt band width under volatility clustering in this sample."
    else:
        vol_text = "The additional volatility model does not dominate standard BB here; estimation error may offset the benefit."

    by_dataset = (
        test_results.groupby("test_dataset")["test_sharpe"]
        .mean()
        .sort_values(ascending=False)
    )
    session_text = "Average Sharpe by test dataset: " + ", ".join(
        f"{idx}={value:.3f}" for idx, value in by_dataset.items()
    )
    if by_dataset.index[0] == "day_test":
        session_text += ". Day session is stronger on average; night may have lower liquidity or higher execution noise."
    elif by_dataset.index[0] == "night_test":
        session_text += ". Night session is stronger on average in this run, suggesting overnight moves may carry usable band signals."
    else:
        session_text += ". Original mixed-session test is strongest on average."
    return vol_text + "\n\n" + session_text

def write_report(
    output_dir: Path,
    data_summary: pd.DataFrame,
    selected_params: pd.DataFrame,
    test_results: pd.DataFrame,
    cost_points_per_side: float,
) -> None:
    best, direction_text = summarize_validation_choice(selected_params)
    selected_display = selected_params[
        [
            "variant",
            "data_used_for_params",
            "bar_minutes",
            "ma_window",
            "k",
            "max_hold_bars",
            "volume_window",
            "vol_ratio",
            "validation_sharpe",
            "turnover",
        ]
    ]
    test_display = test_results[
        [
            "variant",
            "data_used_for_params",
            "test_dataset",
            "bar_minutes",
            "ma_window",
            "k",
            "test_return",
            "test_sharpe",
            "max_dd",
            "turnover",
        ]
    ].sort_values(["variant", "data_used_for_params", "test_dataset"])

    report = f"""# GARCH-Bollinger Strategy First Four Sections / 前四部分草稿

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

{markdown_table(data_summary, float_digits=4)}

Selected parameters and validation performance:

{markdown_table(selected_display, float_digits=4)}

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

`cost_points_per_side = {cost_points_per_side}`. Sharpe uses dynamic `periods_per_year`, so 5/15/20-minute tests are not annualized with a 1-minute factor.

Test results below are final out-of-sample evaluations only. They are not used to choose the strategy or parameters.

{markdown_table(test_display, float_digits=4)}

Primary model selected by validation Sharpe:

```text
variant = {best["variant"]}
data_used_for_params = {best["data_used_for_params"]}
validation_return = {best["validation_return"]:.6f}
validation_sharpe = {best["validation_sharpe"]:.6f}
validation_max_dd = {best["max_dd"]:.6f}
turnover = {best["turnover"]:.2f}
```

Interpretation / 结果解释:

{direction_text}

The test table is then used to report out-of-sample behavior of the validation-selected parameter sets. We do not choose the model by test Sharpe, because that would be data snooping.

{comparison_text(test_results)}

Transaction cost effect: every position change pays `cost_points_per_side / close_(t-1)` on each side through turnover. Therefore high-turnover band settings can show attractive raw timing but weaker net ratio returns after costs. This is why validation selection uses Sharpe first, then less severe drawdown, then lower turnover.

Short comparison with Alexander backup: Alexander filter is a price-filter trend-following backup and does not explicitly model volatility clustering. GARCH-Bollinger is more interpretable for the course discussion because it directly connects Lec4/5 Bollinger Bands with Lec6 GARCH conditional volatility and session-specific market structure. If Alexander performs better in a table, it can be presented as a simpler robust benchmark; if GARCH-BB performs better, the explanation is that dynamic volatility width added useful adaptation.
"""
    (output_dir / "garch_bollinger_first4_report.md").write_text(report, encoding="utf-8")

def parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())

def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GARCH-Bollinger first-four-section experiment using processed split data."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/garch_bollinger_first4")
    parser.add_argument("--bar-minutes", default=",".join(str(v) for v in DEFAULT_BAR_MINUTES))
    parser.add_argument("--ma-windows", default=",".join(str(v) for v in DEFAULT_MA_WINDOWS))
    parser.add_argument("--k-values", default=",".join(str(v) for v in DEFAULT_K_VALUES))
    parser.add_argument("--max-hold-bars", default=",".join(str(v) for v in DEFAULT_MAX_HOLD_BARS))
    parser.add_argument("--volume-windows", default=",".join(str(v) for v in DEFAULT_VOLUME_WINDOWS))
    parser.add_argument("--vol-ratios", default=",".join(str(v) for v in DEFAULT_VOL_RATIOS))
    parser.add_argument("--cost-points-per-side", type=float, default=DEFAULT_COST_POINTS_PER_SIDE)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bar_minutes_values = parse_int_tuple(args.bar_minutes)
    ma_windows = parse_int_tuple(args.ma_windows)
    k_values = parse_float_tuple(args.k_values)
    max_hold_values = parse_int_tuple(args.max_hold_bars)
    volume_windows = parse_int_tuple(args.volume_windows)
    vol_ratios = parse_float_tuple(args.vol_ratios)

    frames = {
        f"{dataset}_{split}": load_processed_split(data_dir, dataset, split)
        for dataset in DATASETS
        for split in SPLITS
    }
    data_summary = write_data_summary(frames, output_dir, DATASETS, SPLITS)
    bars = prepare_bars(frames, bar_minutes_values, DATASETS, SPLITS)
    garch_models = estimate_garch_models(bars, bar_minutes_values, output_dir, DATASETS)

    validation_grid = evaluate_validation_grid(
        bars,
        garch_models,
        bar_minutes_values,
        ma_windows,
        k_values,
        max_hold_values,
        volume_windows,
        vol_ratios,
        args.cost_points_per_side,
        DATASETS,
        VARIANTS,
    )
    validation_grid.to_csv(output_dir / "garch_bollinger_validation_grid.csv", index=False)

    selected_params = select_params(validation_grid, DATASETS, VARIANTS)
    selected_params.to_csv(output_dir / "garch_bollinger_selected_params.csv", index=False)
    selected_params.to_csv(output_dir / "garch_bollinger_validation_selected_results.csv", index=False)

    core_6_selected = select_core_6_params(selected_params)
    core_6_selected.to_csv(output_dir / "garch_bollinger_core_6_validation_selected.csv", index=False)

    test_results = run_test_combinations(
        bars,
        garch_models,
        selected_params,
        output_dir,
        args.cost_points_per_side,
        VARIANTS,
    )
    test_results.to_csv(output_dir / "garch_bollinger_test_results.csv", index=False)

    core_6_results = run_core_6_tests(
        bars,
        garch_models,
        core_6_selected,
        output_dir,
        args.cost_points_per_side,
    )
    core_6_results.to_csv(output_dir / "garch_bollinger_core_6_test_results.csv", index=False)

    original_family_params = select_original_family_params(selected_params)
    original_family_params.to_csv(
        output_dir / "garch_bollinger_original_validbest_day_night_selected.csv",
        index=False,
    )
    original_day_night_results = run_original_best_on_day_night_tests(
        bars,
        garch_models,
        original_family_params,
        output_dir,
        args.cost_points_per_side,
    )
    original_day_night_results.to_csv(
        output_dir / "garch_bollinger_original_validbest_day_night_test_results.csv",
        index=False,
    )

    directional_params = select_directional_family_params(selected_params)
    directional_params.to_csv(
        output_dir / "garch_bollinger_directional_10_validation_selected.csv",
        index=False,
    )
    directional_results = run_directional_10_tests(
        bars,
        garch_models,
        directional_params,
        output_dir,
        args.cost_points_per_side,
    )
    directional_results.to_csv(
        output_dir / "garch_bollinger_directional_10_test_results.csv",
        index=False,
    )

    write_report(output_dir, data_summary, selected_params, test_results, args.cost_points_per_side)

    display_cols = [
        "variant",
        "data_used_for_params",
        "test_dataset",
        "bar_minutes",
        "ma_window",
        "k",
        "test_return",
        "test_sharpe",
        "max_dd",
        "turnover",
    ]
    print("GARCH-Bollinger first-four-section experiment finished")
    print("=" * 72)
    print(f"output_dir: {output_dir}")
    print("Validation-selected parameters:")
    print(
        selected_params.sort_values("validation_sharpe", ascending=False)[
            [
                "variant",
                "data_used_for_params",
                "bar_minutes",
                "ma_window",
                "k",
                "validation_return",
                "validation_sharpe",
                "max_dd",
                "turnover",
            ]
        ].to_string(index=False)
    )
    print("\nCore 6 test evaluations:")
    print(
        core_6_results.sort_values(["data_used_for_params", "strategy_family"])[
            ["strategy_family", *display_cols]
        ].to_string(index=False)
    )
    print("\nOriginal validation-best parameters on day/night tests:")
    print(
        original_day_night_results.sort_values(["strategy_family", "test_dataset"])[
            ["strategy_family", *display_cols]
        ].to_string(index=False)
    )
    print("\nDirectional 10 test evaluations:")
    print(
        directional_results.sort_values(["data_used_for_params", "test_dataset", "strategy_family"])[
            ["strategy_family", *display_cols]
        ].to_string(index=False)
    )
    print("\nFinal test evaluations of validation-selected parameters:")
    print(test_results.sort_values(["variant", "data_used_for_params", "test_dataset"])[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
