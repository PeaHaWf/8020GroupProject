"""GARCH, band construction, volume filters, and position signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float
    alpha_plus_beta: float
    unconditional_vol: float
    omega_scaled: float
    mean_scaled: float
    n_obs: int
    converged: bool
    optimizer_starts: int

def returns_for_garch(df: pd.DataFrame) -> np.ndarray:
    returns = df.loc[~df["segment_start"].astype(bool), "close_return"].astype(float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns.to_numpy(dtype=float)

def garch_variance_scaled(y: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    denom = max(1.0 - alpha - beta, 1e-8)
    unconditional_var = max(omega / denom, 1e-12)
    if len(y) == 0:
        return np.array([], dtype=float)

    u = np.empty_like(y, dtype=float)
    u[0] = unconditional_var
    if len(y) > 1:
        u[1:] = omega + alpha * y[:-1] ** 2
    variance = lfilter([1.0], [1.0, -beta], u)
    return np.maximum(variance, 1e-12)

def fit_garch_qmle(returns: np.ndarray) -> GarchParams:
    y = np.asarray(returns, dtype=float)
    y = y[np.isfinite(y)] * 100.0
    if len(y) < 20:
        return GarchParams(
            omega=1e-12,
            alpha=0.05,
            beta=0.90,
            alpha_plus_beta=0.95,
            unconditional_vol=1e-6,
            omega_scaled=1e-8,
            mean_scaled=0.0,
            n_obs=len(y),
            converged=False,
            optimizer_starts=0,
        )

    mean_scaled = float(np.mean(y))
    y = y - mean_scaled
    sample_var = float(np.var(y))
    sample_var = max(sample_var, 1e-10)
    starts = [
        np.array([sample_var * 0.05, 0.05, 0.90], dtype=float),
        np.array([sample_var * 0.10, 0.10, 0.80], dtype=float),
        np.array([sample_var * 0.02, 0.03, 0.94], dtype=float),
        np.array([sample_var * 0.01, 0.08, 0.88], dtype=float),
    ]
    bounds = (
        (1e-12, max(sample_var * 10.0, 1e-8)),
        (1e-8, 0.60),
        (1e-8, 0.998),
    )
    constraints = ({"type": "ineq", "fun": lambda p: 0.998999 - p[1] - p[2]},)

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e100
        variance = garch_variance_scaled(y, float(omega), float(alpha), float(beta))
        return float(0.5 * np.sum(np.log(variance) + (y**2 / variance)))

    best_result = None
    best_value = float("inf")
    attempted_starts = 0
    for initial in starts:
        attempted_starts += 1
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )
        if np.isfinite(result.fun) and result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is not None and best_result.success:
        omega_scaled, alpha, beta = [float(v) for v in best_result.x]
        converged = True
    else:
        omega_scaled, alpha, beta = [float(v) for v in starts[0]]
        converged = False

    alpha_plus_beta = alpha + beta
    unconditional_var_scaled = omega_scaled / max(1.0 - alpha_plus_beta, 1e-8)
    unconditional_vol = float(np.sqrt(max(unconditional_var_scaled, 1e-12)) / 100.0)
    return GarchParams(
        omega=float(omega_scaled / 10000.0),
        alpha=alpha,
        beta=beta,
        alpha_plus_beta=alpha_plus_beta,
        unconditional_vol=unconditional_vol,
        omega_scaled=omega_scaled,
        mean_scaled=mean_scaled,
        n_obs=len(y),
        converged=converged,
        optimizer_starts=attempted_starts,
    )

def estimate_garch_models(
    bars: dict[tuple[str, str, int], pd.DataFrame],
    bar_minutes_values: tuple[int, ...],
    output_dir: Path,
    datasets: tuple[str, ...],
) -> dict[tuple[str, str, int], GarchParams]:
    models: dict[tuple[str, str, int], GarchParams] = {}
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for fit_sample in ("train", "train_validation"):
            for bar_minutes in bar_minutes_values:
                returns = returns_for_garch(bars[(dataset, fit_sample, bar_minutes)])
                params = fit_garch_qmle(returns)
                models[(dataset, fit_sample, bar_minutes)] = params
                rows.append(
                    {
                        "data_used_for_params": dataset,
                        "fit_sample": fit_sample,
                        "bar_minutes": bar_minutes,
                        "omega": params.omega,
                        "alpha": params.alpha,
                        "beta": params.beta,
                        "alpha_plus_beta": params.alpha_plus_beta,
                        "unconditional_vol": params.unconditional_vol,
                        "mean_scaled": params.mean_scaled,
                        "n_obs": params.n_obs,
                        "converged": params.converged,
                        "optimizer_starts": params.optimizer_starts,
                    }
                )
    model_params = pd.DataFrame(rows)
    model_params.to_csv(output_dir / "garch_model_params.csv", index=False)
    return models

def add_standard_bands(df: pd.DataFrame, ma_window: int, k: float) -> pd.DataFrame:
    out = df.copy()
    out["ma"] = np.nan
    out["rolling_std"] = np.nan
    for _, group in out.groupby("segment_id", sort=False):
        idx = group.index
        close = group["hi1_close"].astype(float)
        out.loc[idx, "ma"] = close.rolling(ma_window, min_periods=ma_window).mean().shift(1)
        out.loc[idx, "rolling_std"] = close.rolling(ma_window, min_periods=ma_window).std(ddof=1).shift(1)
    out["sigma"] = np.nan
    out["upper_band"] = out["ma"] + k * out["rolling_std"]
    out["lower_band"] = out["ma"] - k * out["rolling_std"]
    return out

def add_garch_bands(df: pd.DataFrame, ma_window: int, k: float, params: GarchParams) -> pd.DataFrame:
    out = df.copy()
    out["ma"] = np.nan
    out["sigma"] = np.nan
    for _, group in out.groupby("segment_id", sort=False):
        idx = group.index
        close = group["hi1_close"].astype(float)
        returns = group["close_return"].astype(float).fillna(0.0).to_numpy(dtype=float)
        y = returns * 100.0 - params.mean_scaled
        variance_scaled = garch_variance_scaled(y, params.omega_scaled, params.alpha, params.beta)
        out.loc[idx, "ma"] = close.rolling(ma_window, min_periods=ma_window).mean().shift(1)
        out.loc[idx, "sigma"] = np.sqrt(variance_scaled) / 100.0

    previous_close = out.groupby("segment_id", sort=False)["hi1_close"].shift(1).astype(float)
    band_width = k * previous_close * out["sigma"].astype(float)
    out["upper_band"] = out["ma"] + band_width
    out["lower_band"] = out["ma"] - band_width
    return out

def add_volume_confirmation(df: pd.DataFrame, volume_window: int, vol_ratio: float) -> pd.DataFrame:
    out = df.copy()
    out["volume_ma"] = np.nan
    for _, group in out.groupby("segment_id", sort=False):
        idx = group.index
        volume = group["hi1_volume"].astype(float)
        out.loc[idx, "volume_ma"] = volume.rolling(volume_window, min_periods=volume_window).mean().shift(1)
    out["volume_ok"] = out["hi1_volume"].astype(float) > vol_ratio * out["volume_ma"].astype(float)
    out["volume_ok"] = out["volume_ok"].fillna(False)
    return out

def variant_uses_garch(variant: str) -> bool:
    return variant.startswith("garch_")

def variant_uses_volume(variant: str) -> bool:
    return variant.endswith("_volume")

def variant_direction(variant: str) -> str:
    if "momentum" in variant:
        return "momentum"
    return "contrarian"

def signal_from_bands(close: float, upper: float, lower: float, direction: str) -> int:
    if not np.isfinite(close) or not np.isfinite(upper) or not np.isfinite(lower):
        return 0
    if close > upper:
        return 1 if direction == "momentum" else -1
    if close < lower:
        return -1 if direction == "momentum" else 1
    return 0

def generate_target_positions(
    df: pd.DataFrame,
    variant: str,
    max_hold_bars: int,
) -> pd.Series:
    direction = variant_direction(variant)
    use_volume = variant_uses_volume(variant)
    positions = pd.Series(0, index=df.index, name="target_position", dtype="int64")

    for _, group in df.groupby("segment_id", sort=False):
        current_position = 0
        hold_count = 0
        group_positions: list[int] = []
        index_values = group.index.to_numpy()
        closes = group["hi1_close"].to_numpy(dtype=float)
        ma_values = group["ma"].to_numpy(dtype=float)
        upper_values = group["upper_band"].to_numpy(dtype=float)
        lower_values = group["lower_band"].to_numpy(dtype=float)
        if use_volume and "volume_ok" in group.columns:
            volume_values = group["volume_ok"].to_numpy(dtype=bool)
        else:
            volume_values = np.ones(len(group), dtype=bool)

        for row_number in range(len(group)):
            if row_number == 0:
                current_position = 0
                hold_count = 0
                group_positions.append(0)
                continue

            close = closes[row_number]
            ma = ma_values[row_number]
            upper = upper_values[row_number]
            lower = lower_values[row_number]
            volume_ok = bool(volume_values[row_number])
            entry_signal = signal_from_bands(close, upper, lower, direction)

            if direction == "contrarian":
                if current_position == 1 and np.isfinite(ma) and close >= ma:
                    current_position = 0
                    hold_count = 0
                elif current_position == -1 and np.isfinite(ma) and close <= ma:
                    current_position = 0
                    hold_count = 0

                if entry_signal != 0 and entry_signal != current_position:
                    if volume_ok:
                        current_position = entry_signal
                        hold_count = 0
                    elif current_position != 0 and entry_signal == -current_position:
                        current_position = 0
                        hold_count = 0

            else:
                if current_position != 0 and hold_count >= max_hold_bars:
                    current_position = 0
                    hold_count = 0

                if entry_signal != 0 and entry_signal != current_position:
                    if volume_ok:
                        current_position = entry_signal
                        hold_count = 0
                    elif current_position != 0 and entry_signal == -current_position:
                        current_position = 0
                        hold_count = 0

            group_positions.append(current_position)
            if current_position != 0:
                hold_count += 1
            else:
                hold_count = 0

        if group_positions:
            group_positions[-1] = 0
        positions.loc[index_values] = np.asarray(group_positions, dtype=np.int64)

    return positions

