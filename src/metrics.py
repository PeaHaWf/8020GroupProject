from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MarketConfig


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    values = np.asarray(equity, dtype=float)
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = values / np.maximum(running_max, 1e-12) - 1.0
    return float(drawdowns.min())


def log_return(equity: pd.Series | np.ndarray) -> float:
    values = np.asarray(equity, dtype=float)
    if values.size < 2:
        return 0.0
    if values[0] <= 0 or values[-1] <= 0:
        return -1_000_000.0
    return float(np.log(values[-1] / values[0]))


def sharpe_ratio(pnl: pd.Series | np.ndarray, bars_per_year: int) -> float:
    values = np.asarray(pnl, dtype=float)
    if values.size < 2:
        return 0.0
    std = values.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(values.mean() / std * np.sqrt(bars_per_year))


def evaluate_trace(trace: pd.DataFrame, market_config: MarketConfig | None = None) -> dict[str, float]:
    cfg = market_config or MarketConfig()
    if trace.empty:
        return {
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "log_return": 0.0,
            "total_pnl": 0.0,
            "final_equity": cfg.initial_capital,
            "turnover": 0.0,
        }

    return {
        "max_drawdown": max_drawdown(trace["equity"]),
        "sharpe_ratio": sharpe_ratio(trace["net_pnl"], cfg.bars_per_year),
        "log_return": log_return(trace["equity"]),
        "total_pnl": float(trace["net_pnl"].sum()),
        "final_equity": float(trace["equity"].iloc[-1]),
        "turnover": float(trace["trade_size"].sum()),
    }
