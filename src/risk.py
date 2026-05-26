from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import MarketConfig, dataset_path, model_path
from .evaluation import run_policy_on_dataframe
from .metrics import evaluate_trace


def trade_ledger_from_trace(trace: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    trades = []
    active_position = 0
    entry_time = None
    entry_price = None
    cumulative_pnl = 0.0

    for _, row in trace.iterrows():
        position = int(row["position"])
        previous_position = int(row["previous_position"])
        changed = position != previous_position
        cumulative_pnl += float(row["net_pnl"])

        if changed and previous_position != 0:
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": row["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": row["price"],
                    "position": active_position,
                    "trade_pnl": cumulative_pnl,
                    "balance_after_trade": float(row["equity"]),
                }
            )
            cumulative_pnl = 0.0

        if changed and position != 0:
            active_position = position
            entry_time = row["timestamp"]
            entry_price = row["price"]
            cumulative_pnl = 0.0

    if active_position != 0 and not trace.empty:
        last = trace.iloc[-1]
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": last["timestamp"],
                "entry_price": entry_price,
                "exit_price": last["price"],
                "position": active_position,
                "trade_pnl": cumulative_pnl,
                "balance_after_trade": float(last["equity"]),
            }
        )

    ledger = pd.DataFrame(trades)
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "position",
                "trade_pnl",
                "balance_after_trade",
            ]
        )
    ledger.insert(0, "trade_id", range(1, len(ledger) + 1))
    ledger["initial_capital"] = initial_capital
    return ledger


def run_risk_management(output_dir: Path, market_config: MarketConfig | None = None) -> dict:
    cfg = market_config or MarketConfig()
    best_path = output_dir / "best_model.json"
    if not best_path.exists():
        raise FileNotFoundError("best_model.json not found; run evaluation before risk management")

    best = json.loads(best_path.read_text(encoding="utf-8"))
    model_name = str(best["model"]).replace("_model", "")
    dataset_name = str(best["dataset"]).replace("_test", "")
    test_df = pd.read_csv(dataset_path(dataset_name, "test"), parse_dates=["timestamp"])
    trace = run_policy_on_dataframe(model_path(model_name), test_df, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "risk_equity_curve.csv"
    trades_path = output_dir / "risk_trade_ledger.csv"
    summary_path = output_dir / "risk_summary.json"

    trace.to_csv(equity_path, index=False)
    ledger = trade_ledger_from_trace(trace, cfg.initial_capital)
    ledger.to_csv(trades_path, index=False)

    summary = {
        "selected_model": best["model"],
        "selected_dataset": best["dataset"],
        "initial_capital": cfg.initial_capital,
        "metrics": evaluate_trace(trace, cfg),
        "equity_curve_path": str(equity_path),
        "trade_ledger_path": str(trades_path),
        "number_of_trades": int(len(ledger)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
