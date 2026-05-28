"""Backtesting, validation selection, and test evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from garch_bollinger_strategy import (
    GarchParams,
    add_garch_bands,
    add_standard_bands,
    add_volume_confirmation,
    generate_target_positions,
    variant_direction,
    variant_uses_garch,
    variant_uses_volume,
)


def backtest_non_capital(
    df: pd.DataFrame,
    target_position: pd.Series,
    cost_points_per_side: float,
) -> pd.DataFrame:
    out = df.copy()
    out["target_position"] = target_position.reindex(out.index).fillna(0).astype(int)

    out["executed_position"] = (
        out.groupby("segment_id", sort=False)["target_position"].shift(1).fillna(0).astype(int)
    )
    out.loc[out["segment_start"].astype(bool), "executed_position"] = 0
    out["close_return"] = out["close_return"].astype(float).fillna(0.0)
    out.loc[out["segment_start"].astype(bool), "close_return"] = 0.0

    previous_executed = out.groupby("segment_id", sort=False)["executed_position"].shift(1).fillna(0)
    out["turnover"] = (out["executed_position"] - previous_executed).abs().astype(float)
    previous_close = out.groupby("segment_id", sort=False)["hi1_close"].shift(1).astype(float)
    out["cost_return"] = (
        out["turnover"] * cost_points_per_side / previous_close
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["strategy_return"] = out["executed_position"] * out["close_return"] - out["cost_return"]
    out["cumulative_return"] = (1.0 + out["strategy_return"]).cumprod() - 1.0
    return out

def estimate_periods_per_year(result: pd.DataFrame) -> float:
    bars_per_day = result.groupby("trading_date").size()
    if bars_per_day.empty:
        return 0.0
    return float(bars_per_day.mean() * 252)

def calculate_metrics(result: pd.DataFrame) -> dict[str, float]:
    returns = result["strategy_return"].astype(float)
    cumulative_curve = 1.0 + result["cumulative_return"].astype(float)
    periods_per_year = estimate_periods_per_year(result)
    total_return = float((1.0 + returns).prod() - 1.0)
    log_return = float(np.log1p(total_return)) if total_return > -1 else float("-inf")

    std = returns.std(ddof=1)
    sharpe = 0.0
    if std > 1e-12 and periods_per_year > 0:
        sharpe = float(returns.mean() / std * np.sqrt(periods_per_year))

    running_peak = cumulative_curve.cummax()
    max_dd = float((cumulative_curve / running_peak - 1.0).min()) if not cumulative_curve.empty else 0.0
    return {
        "return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "log_return": log_return,
        "turnover": float(result["turnover"].sum()),
        "periods_per_year": periods_per_year,
    }

def evaluate_one(
    bars: pd.DataFrame,
    variant: str,
    ma_window: int,
    k: float,
    max_hold_bars: int,
    volume_window: int | None,
    vol_ratio: float | None,
    garch_params: GarchParams | None,
    cost_points_per_side: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if variant_uses_garch(variant):
        if garch_params is None:
            raise ValueError(f"{variant} requires GARCH parameters")
        prepared = add_garch_bands(bars, ma_window, k, garch_params)
    else:
        prepared = add_standard_bands(bars, ma_window, k)

    if variant_uses_volume(variant):
        if volume_window is None or vol_ratio is None:
            raise ValueError(f"{variant} requires volume parameters")
        prepared = add_volume_confirmation(prepared, volume_window, vol_ratio)
    else:
        prepared["volume_ma"] = np.nan
        prepared["volume_ok"] = True

    target_position = generate_target_positions(prepared, variant, max_hold_bars)
    result = backtest_non_capital(prepared, target_position, cost_points_per_side)
    return result, calculate_metrics(result)

def evaluate_prepared(
    prepared: pd.DataFrame,
    variant: str,
    max_hold_bars: int,
    cost_points_per_side: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    target_position = generate_target_positions(prepared, variant, max_hold_bars)
    result = backtest_non_capital(prepared, target_position, cost_points_per_side)
    return result, calculate_metrics(result)

def iter_parameter_grid(
    variant: str,
    bar_minutes_values: tuple[int, ...],
    ma_windows: tuple[int, ...],
    k_values: tuple[float, ...],
    max_hold_values: tuple[int, ...],
    volume_windows: tuple[int, ...],
    vol_ratios: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    direction = variant_direction(variant)
    use_volume = variant_uses_volume(variant)
    hold_values = max_hold_values if direction == "momentum" else (0,)
    volume_window_values = volume_windows if use_volume else (None,)
    vol_ratio_values = vol_ratios if use_volume else (None,)

    for bar_minutes in bar_minutes_values:
        for ma_window in ma_windows:
            for k in k_values:
                for max_hold_bars in hold_values:
                    for volume_window in volume_window_values:
                        for vol_ratio in vol_ratio_values:
                            rows.append(
                                {
                                    "bar_minutes": int(bar_minutes),
                                    "ma_window": int(ma_window),
                                    "k": float(k),
                                    "max_hold_bars": int(max_hold_bars),
                                    "volume_window": int(volume_window) if volume_window is not None else np.nan,
                                    "vol_ratio": float(vol_ratio) if vol_ratio is not None else np.nan,
                                }
                            )
    return rows

def evaluate_validation_grid(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    garch_models: dict[tuple[str, str, int], GarchParams],
    bar_minutes_values: tuple[int, ...],
    ma_windows: tuple[int, ...],
    k_values: tuple[float, ...],
    max_hold_values: tuple[int, ...],
    volume_windows: tuple[int, ...],
    vol_ratios: tuple[float, ...],
    cost_points_per_side: float,
    datasets: tuple[str, ...],
    variants: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for variant in variants:
            print(f"validation grid: dataset={dataset}, variant={variant}", flush=True)
            band_cache: dict[tuple[object, ...], pd.DataFrame] = {}
            volume_cache: dict[tuple[object, ...], pd.DataFrame] = {}
            params_grid = iter_parameter_grid(
                variant,
                bar_minutes_values,
                ma_windows,
                k_values,
                max_hold_values,
                volume_windows,
                vol_ratios,
            )
            for params in params_grid:
                bar_minutes = int(params["bar_minutes"])
                garch_params = (
                    garch_models[(dataset, "train", bar_minutes)] if variant_uses_garch(variant) else None
                )
                band_key = (
                    "garch" if variant_uses_garch(variant) else "standard",
                    bar_minutes,
                    int(params["ma_window"]),
                    float(params["k"]),
                )
                if band_key not in band_cache:
                    base_bars = bars[(dataset, "validation", bar_minutes)]
                    if variant_uses_garch(variant):
                        if garch_params is None:
                            raise ValueError(f"{variant} requires GARCH parameters")
                        band_cache[band_key] = add_garch_bands(
                            base_bars,
                            int(params["ma_window"]),
                            float(params["k"]),
                            garch_params,
                        )
                    else:
                        band_cache[band_key] = add_standard_bands(
                            base_bars,
                            int(params["ma_window"]),
                            float(params["k"]),
                        )

                if variant_uses_volume(variant):
                    volume_window = int(params["volume_window"])
                    vol_ratio = float(params["vol_ratio"])
                    volume_key = (*band_key, volume_window, vol_ratio)
                    if volume_key not in volume_cache:
                        volume_cache[volume_key] = add_volume_confirmation(
                            band_cache[band_key],
                            volume_window,
                            vol_ratio,
                        )
                    prepared = volume_cache[volume_key]
                else:
                    no_volume_key = (*band_key, "no_volume")
                    if no_volume_key not in volume_cache:
                        prepared_no_volume = band_cache[band_key].copy()
                        prepared_no_volume["volume_ma"] = np.nan
                        prepared_no_volume["volume_ok"] = True
                        volume_cache[no_volume_key] = prepared_no_volume
                    prepared = volume_cache[no_volume_key]

                _, metrics = evaluate_prepared(
                    prepared,
                    variant,
                    int(params["max_hold_bars"]),
                    cost_points_per_side,
                )
                model_params = garch_params
                rows.append(
                    {
                        "variant": variant,
                        "data_used_for_params": dataset,
                        **params,
                        "omega": model_params.omega if model_params else np.nan,
                        "alpha": model_params.alpha if model_params else np.nan,
                        "beta": model_params.beta if model_params else np.nan,
                        "alpha_plus_beta": model_params.alpha_plus_beta if model_params else np.nan,
                        "unconditional_vol": model_params.unconditional_vol if model_params else np.nan,
                        "validation_return": metrics["return"],
                        "validation_sharpe": metrics["sharpe"],
                        "max_dd": metrics["max_dd"],
                        "log_return": metrics["log_return"],
                        "turnover": metrics["turnover"],
                        "periods_per_year": metrics["periods_per_year"],
                    }
                )
    return pd.DataFrame(rows)

def select_params(
    validation_grid: pd.DataFrame,
    datasets: tuple[str, ...],
    variants: tuple[str, ...],
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for dataset in datasets:
        for variant in variants:
            subset = validation_grid.loc[
                validation_grid["data_used_for_params"].eq(dataset)
                & validation_grid["variant"].eq(variant)
            ].copy()
            nonzero = subset.loc[subset["turnover"].gt(0)]
            candidates = nonzero if not nonzero.empty else subset
            best = candidates.sort_values(
                ["validation_sharpe", "max_dd", "turnover"],
                ascending=[False, False, True],
            ).iloc[0]
            selected_rows.append(best)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected[
        [
            "variant",
            "data_used_for_params",
            "bar_minutes",
            "ma_window",
            "k",
            "max_hold_bars",
            "volume_window",
            "vol_ratio",
            "omega",
            "alpha",
            "beta",
            "alpha_plus_beta",
            "unconditional_vol",
            "validation_return",
            "validation_sharpe",
            "max_dd",
            "log_return",
            "turnover",
            "periods_per_year",
        ]
    ]

def safe_trace_name(variant: str, param_source: str, test_dataset: str) -> str:
    return f"trace_{variant}_{param_source}_params_on_{test_dataset}_test.csv"

def run_test_combinations(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    garch_models: dict[tuple[str, str, int], GarchParams],
    selected_params: pd.DataFrame,
    output_dir: Path,
    cost_points_per_side: float,
    variants: tuple[str, ...],
) -> pd.DataFrame:
    combos = [
        ("original", "original"),
        ("original", "day"),
        ("original", "night"),
        ("day", "day"),
        ("night", "night"),
    ]
    rows: list[dict[str, object]] = []

    for variant in variants:
        for param_source, test_dataset in combos:
            selected = selected_params.loc[
                selected_params["variant"].eq(variant)
                & selected_params["data_used_for_params"].eq(param_source)
            ].iloc[0]
            bar_minutes = int(selected["bar_minutes"])
            ma_window = int(selected["ma_window"])
            k = float(selected["k"])
            max_hold_bars = int(selected["max_hold_bars"])
            volume_window = None if pd.isna(selected["volume_window"]) else int(selected["volume_window"])
            vol_ratio = None if pd.isna(selected["vol_ratio"]) else float(selected["vol_ratio"])
            garch_params = (
                garch_models[(param_source, "train_validation", bar_minutes)]
                if variant_uses_garch(variant)
                else None
            )
            trace, metrics = evaluate_one(
                bars[(test_dataset, "test", bar_minutes)],
                variant,
                ma_window,
                k,
                max_hold_bars,
                volume_window,
                vol_ratio,
                garch_params,
                cost_points_per_side,
            )

            trace_path = output_dir / safe_trace_name(variant, param_source, test_dataset)
            trace_cols = [
                "timestamp",
                "trading_date",
                "hi1_close",
                "close_return",
                "ma",
                "sigma",
                "upper_band",
                "lower_band",
                "target_position",
                "executed_position",
                "turnover",
                "cost_return",
                "strategy_return",
                "cumulative_return",
            ]
            trace_out = trace[trace_cols].rename(columns={"hi1_close": "close"})
            trace_out.to_csv(trace_path, index=False)

            rows.append(
                {
                    "variant": variant,
                    "data_used_for_params": param_source,
                    "test_dataset": f"{test_dataset}_test",
                    "bar_minutes": bar_minutes,
                    "ma_window": ma_window,
                    "k": k,
                    "max_hold_bars": max_hold_bars,
                    "volume_window": volume_window if volume_window is not None else np.nan,
                    "vol_ratio": vol_ratio if vol_ratio is not None else np.nan,
                    "omega": garch_params.omega if garch_params else np.nan,
                    "alpha": garch_params.alpha if garch_params else np.nan,
                    "beta": garch_params.beta if garch_params else np.nan,
                    "alpha_plus_beta": garch_params.alpha_plus_beta if garch_params else np.nan,
                    "test_return": metrics["return"],
                    "test_sharpe": metrics["sharpe"],
                    "max_dd": metrics["max_dd"],
                    "log_return": metrics["log_return"],
                    "turnover": metrics["turnover"],
                    "periods_per_year": metrics["periods_per_year"],
                    "trace_path": str(trace_path),
                }
            )
    return pd.DataFrame(rows)



def select_core_6_params(selected_params: pd.DataFrame) -> pd.DataFrame:
    """Select 6 validation winners: dataset x {momentum, contrarian}."""
    rows: list[pd.Series] = []
    family_variants = {
        "momentum": ("garch_bb_momentum", "garch_bb_momentum_volume"),
        "contrarian": (
            "standard_bb_contrarian",
            "garch_bb_contrarian",
            "garch_bb_contrarian_volume",
        ),
    }
    datasets = ("original", "day", "night")

    for dataset in datasets:
        for family, variants in family_variants.items():
            subset = selected_params.loc[
                selected_params["data_used_for_params"].eq(dataset)
                & selected_params["variant"].isin(variants)
            ].copy()
            if subset.empty:
                raise ValueError(f"No selected validation rows for {dataset}/{family}")
            best = subset.sort_values(
                ["validation_sharpe", "validation_return", "max_dd", "turnover"],
                ascending=[False, False, False, True],
            ).iloc[0].copy()
            best["strategy_family"] = family
            rows.append(best)

    cols = ["strategy_family"] + [c for c in selected_params.columns if c in rows[0].index]
    return pd.DataFrame(rows).reset_index(drop=True)[cols]


def run_core_6_tests(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    garch_models: dict[tuple[str, str, int], GarchParams],
    core_6_params: pd.DataFrame,
    output_dir: Path,
    cost_points_per_side: float,
) -> pd.DataFrame:
    """Evaluate each core-6 validation winner on its corresponding test set only."""
    rows: list[dict[str, object]] = []

    for _, selected in core_6_params.iterrows():
        param_source = str(selected["data_used_for_params"])
        test_dataset = param_source
        variant = str(selected["variant"])
        strategy_family = str(selected["strategy_family"])
        bar_minutes = int(selected["bar_minutes"])
        ma_window = int(selected["ma_window"])
        k = float(selected["k"])
        max_hold_bars = int(selected["max_hold_bars"])
        volume_window = None if pd.isna(selected["volume_window"]) else int(selected["volume_window"])
        vol_ratio = None if pd.isna(selected["vol_ratio"]) else float(selected["vol_ratio"])
        garch_params = (
            garch_models[(param_source, "train_validation", bar_minutes)]
            if variant_uses_garch(variant)
            else None
        )

        trace, metrics = evaluate_one(
            bars[(test_dataset, "test", bar_minutes)],
            variant,
            ma_window,
            k,
            max_hold_bars,
            volume_window,
            vol_ratio,
            garch_params,
            cost_points_per_side,
        )

        trace_path = output_dir / (
            f"trace_core6_{strategy_family}_{variant}_{param_source}_params_on_{test_dataset}_test.csv"
        )
        trace_cols = [
            "timestamp",
            "trading_date",
            "hi1_close",
            "close_return",
            "ma",
            "sigma",
            "upper_band",
            "lower_band",
            "target_position",
            "executed_position",
            "turnover",
            "cost_return",
            "strategy_return",
            "cumulative_return",
        ]
        trace_out = trace[trace_cols].rename(columns={"hi1_close": "close"})
        trace_out.to_csv(trace_path, index=False)

        rows.append(
            {
                "strategy_family": strategy_family,
                "variant": variant,
                "data_used_for_params": param_source,
                "test_dataset": f"{test_dataset}_test",
                "bar_minutes": bar_minutes,
                "ma_window": ma_window,
                "k": k,
                "max_hold_bars": max_hold_bars,
                "volume_window": volume_window if volume_window is not None else np.nan,
                "vol_ratio": vol_ratio if vol_ratio is not None else np.nan,
                "omega": garch_params.omega if garch_params else np.nan,
                "alpha": garch_params.alpha if garch_params else np.nan,
                "beta": garch_params.beta if garch_params else np.nan,
                "alpha_plus_beta": garch_params.alpha_plus_beta if garch_params else np.nan,
                "validation_return": float(selected["validation_return"]),
                "validation_sharpe": float(selected["validation_sharpe"]),
                "validation_max_dd": float(selected["max_dd"]),
                "validation_turnover": float(selected["turnover"]),
                "test_return": metrics["return"],
                "test_sharpe": metrics["sharpe"],
                "max_dd": metrics["max_dd"],
                "log_return": metrics["log_return"],
                "turnover": metrics["turnover"],
                "periods_per_year": metrics["periods_per_year"],
                "trace_path": str(trace_path),
            }
        )
    return pd.DataFrame(rows)


def select_original_family_params(selected_params: pd.DataFrame) -> pd.DataFrame:
    """Select original validation winners for momentum and contrarian families."""
    rows: list[pd.Series] = []
    family_variants = {
        "momentum": ("garch_bb_momentum", "garch_bb_momentum_volume"),
        "contrarian": (
            "standard_bb_contrarian",
            "garch_bb_contrarian",
            "garch_bb_contrarian_volume",
        ),
    }

    for family, variants in family_variants.items():
        subset = selected_params.loc[
            selected_params["data_used_for_params"].eq("original")
            & selected_params["variant"].isin(variants)
        ].copy()
        if subset.empty:
            raise ValueError(f"No selected validation rows for original/{family}")
        best = subset.sort_values(
            ["validation_sharpe", "validation_return", "max_dd", "turnover"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()
        best["strategy_family"] = family
        rows.append(best)

    cols = ["strategy_family"] + [c for c in selected_params.columns if c in rows[0].index]
    return pd.DataFrame(rows).reset_index(drop=True)[cols]


def run_original_best_on_day_night_tests(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    garch_models: dict[tuple[str, str, int], GarchParams],
    original_family_params: pd.DataFrame,
    output_dir: Path,
    cost_points_per_side: float,
) -> pd.DataFrame:
    """Run original validation winners on day_test and night_test only."""
    rows: list[dict[str, object]] = []

    for _, selected in original_family_params.iterrows():
        param_source = "original"
        variant = str(selected["variant"])
        strategy_family = str(selected["strategy_family"])
        bar_minutes = int(selected["bar_minutes"])
        ma_window = int(selected["ma_window"])
        k = float(selected["k"])
        max_hold_bars = int(selected["max_hold_bars"])
        volume_window = None if pd.isna(selected["volume_window"]) else int(selected["volume_window"])
        vol_ratio = None if pd.isna(selected["vol_ratio"]) else float(selected["vol_ratio"])
        garch_params = (
            garch_models[(param_source, "train_validation", bar_minutes)]
            if variant_uses_garch(variant)
            else None
        )

        for test_dataset in ("day", "night"):
            trace, metrics = evaluate_one(
                bars[(test_dataset, "test", bar_minutes)],
                variant,
                ma_window,
                k,
                max_hold_bars,
                volume_window,
                vol_ratio,
                garch_params,
                cost_points_per_side,
            )

            trace_path = output_dir / (
                f"trace_original_validbest_{strategy_family}_{variant}_on_{test_dataset}_test.csv"
            )
            trace_cols = [
                "timestamp",
                "trading_date",
                "hi1_close",
                "close_return",
                "ma",
                "sigma",
                "upper_band",
                "lower_band",
                "target_position",
                "executed_position",
                "turnover",
                "cost_return",
                "strategy_return",
                "cumulative_return",
            ]
            trace_out = trace[trace_cols].rename(columns={"hi1_close": "close"})
            trace_out.to_csv(trace_path, index=False)

            rows.append(
                {
                    "strategy_family": strategy_family,
                    "variant": variant,
                    "data_used_for_params": param_source,
                    "test_dataset": f"{test_dataset}_test",
                    "bar_minutes": bar_minutes,
                    "ma_window": ma_window,
                    "k": k,
                    "max_hold_bars": max_hold_bars,
                    "volume_window": volume_window if volume_window is not None else np.nan,
                    "vol_ratio": vol_ratio if vol_ratio is not None else np.nan,
                    "omega": garch_params.omega if garch_params else np.nan,
                    "alpha": garch_params.alpha if garch_params else np.nan,
                    "beta": garch_params.beta if garch_params else np.nan,
                    "alpha_plus_beta": garch_params.alpha_plus_beta if garch_params else np.nan,
                    "validation_return": float(selected["validation_return"]),
                    "validation_sharpe": float(selected["validation_sharpe"]),
                    "validation_max_dd": float(selected["max_dd"]),
                    "validation_turnover": float(selected["turnover"]),
                    "test_return": metrics["return"],
                    "test_sharpe": metrics["sharpe"],
                    "max_dd": metrics["max_dd"],
                    "log_return": metrics["log_return"],
                    "turnover": metrics["turnover"],
                    "periods_per_year": metrics["periods_per_year"],
                    "trace_path": str(trace_path),
                }
            )
    return pd.DataFrame(rows)


def select_directional_family_params(selected_params: pd.DataFrame) -> pd.DataFrame:
    """Select validation winners for each parameter source and direction family."""
    rows: list[pd.Series] = []
    family_variants = {
        "momentum": ("garch_bb_momentum", "garch_bb_momentum_volume"),
        "contrarian": (
            "standard_bb_contrarian",
            "garch_bb_contrarian",
            "garch_bb_contrarian_volume",
        ),
    }

    for param_source in ("original", "day", "night"):
        for family, variants in family_variants.items():
            subset = selected_params.loc[
                selected_params["data_used_for_params"].eq(param_source)
                & selected_params["variant"].isin(variants)
            ].copy()
            if subset.empty:
                raise ValueError(f"No selected validation rows for {param_source}/{family}")
            best = subset.sort_values(
                ["validation_sharpe", "validation_return", "max_dd", "turnover"],
                ascending=[False, False, False, True],
            ).iloc[0].copy()
            best["strategy_family"] = family
            rows.append(best)

    cols = ["strategy_family"] + [c for c in selected_params.columns if c in rows[0].index]
    return pd.DataFrame(rows).reset_index(drop=True)[cols]


def run_directional_10_tests(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    garch_models: dict[tuple[str, str, int], GarchParams],
    directional_params: pd.DataFrame,
    output_dir: Path,
    cost_points_per_side: float,
) -> pd.DataFrame:
    """Run 5 parameter/test combinations for momentum and contrarian families."""
    combos = [
        ("original", "original"),
        ("original", "day"),
        ("original", "night"),
        ("day", "day"),
        ("night", "night"),
    ]
    rows: list[dict[str, object]] = []

    for param_source, test_dataset in combos:
        for strategy_family in ("momentum", "contrarian"):
            selected = directional_params.loc[
                directional_params["data_used_for_params"].eq(param_source)
                & directional_params["strategy_family"].eq(strategy_family)
            ].iloc[0]
            variant = str(selected["variant"])
            bar_minutes = int(selected["bar_minutes"])
            ma_window = int(selected["ma_window"])
            k = float(selected["k"])
            max_hold_bars = int(selected["max_hold_bars"])
            volume_window = None if pd.isna(selected["volume_window"]) else int(selected["volume_window"])
            vol_ratio = None if pd.isna(selected["vol_ratio"]) else float(selected["vol_ratio"])
            garch_params = (
                garch_models[(param_source, "train_validation", bar_minutes)]
                if variant_uses_garch(variant)
                else None
            )

            trace, metrics = evaluate_one(
                bars[(test_dataset, "test", bar_minutes)],
                variant,
                ma_window,
                k,
                max_hold_bars,
                volume_window,
                vol_ratio,
                garch_params,
                cost_points_per_side,
            )

            trace_path = output_dir / (
                f"trace_directional10_{strategy_family}_{variant}_{param_source}_params_on_{test_dataset}_test.csv"
            )
            trace_cols = [
                "timestamp",
                "trading_date",
                "hi1_close",
                "close_return",
                "ma",
                "sigma",
                "upper_band",
                "lower_band",
                "target_position",
                "executed_position",
                "turnover",
                "cost_return",
                "strategy_return",
                "cumulative_return",
            ]
            trace_out = trace[trace_cols].rename(columns={"hi1_close": "close"})
            trace_out.to_csv(trace_path, index=False)

            rows.append(
                {
                    "strategy_family": strategy_family,
                    "variant": variant,
                    "data_used_for_params": param_source,
                    "test_dataset": f"{test_dataset}_test",
                    "bar_minutes": bar_minutes,
                    "ma_window": ma_window,
                    "k": k,
                    "max_hold_bars": max_hold_bars,
                    "volume_window": volume_window if volume_window is not None else np.nan,
                    "vol_ratio": vol_ratio if vol_ratio is not None else np.nan,
                    "omega": garch_params.omega if garch_params else np.nan,
                    "alpha": garch_params.alpha if garch_params else np.nan,
                    "beta": garch_params.beta if garch_params else np.nan,
                    "alpha_plus_beta": garch_params.alpha_plus_beta if garch_params else np.nan,
                    "validation_return": float(selected["validation_return"]),
                    "validation_sharpe": float(selected["validation_sharpe"]),
                    "validation_max_dd": float(selected["max_dd"]),
                    "validation_turnover": float(selected["turnover"]),
                    "test_return": metrics["return"],
                    "test_sharpe": metrics["sharpe"],
                    "max_dd": metrics["max_dd"],
                    "log_return": metrics["log_return"],
                    "turnover": metrics["turnover"],
                    "periods_per_year": metrics["periods_per_year"],
                    "trace_path": str(trace_path),
                }
            )
    return pd.DataFrame(rows)
