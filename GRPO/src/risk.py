from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketConfig, RiskConfig, dataset_path, model_path
from .evaluation import run_policy_on_dataframe
from .metrics import evaluate_trace
from .policy import action_from_policy, load_model
from .trading_env import TradingEnv


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


def _cost_per_contract(market_config: MarketConfig) -> float:
    return market_config.transaction_cost_per_contract + (
        market_config.slippage_points * market_config.contract_multiplier
    )


def _contracts_for_risk(
    equity: float,
    volatility_points: float,
    risk_config: RiskConfig,
    market_config: MarketConfig,
    max_contracts: int,
) -> tuple[int, float]:
    risk_budget = equity * risk_config.risk_per_trade
    point_risk = max(float(volatility_points), risk_config.min_volatility_points)
    risk_per_contract = point_risk * market_config.contract_multiplier
    contracts = int(np.floor(risk_budget / max(risk_per_contract, 1e-12)))
    contracts = max(0, min(max_contracts, contracts))
    return contracts, risk_budget


def _dynamic_trade_ledger(trace: pd.DataFrame) -> pd.DataFrame:
    trades = []
    active = None

    for _, row in trace.iterrows():
        previous_contracts = int(row["previous_contracts"])
        executed_contracts = int(row["executed_contracts"])
        changed = previous_contracts != executed_contracts

        if changed and previous_contracts != 0 and active is not None:
            active["trade_pnl"] += float(row["net_pnl"])
            active["max_adverse_pnl"] = min(active["max_adverse_pnl"], active["trade_pnl"])
            active["max_favorable_pnl"] = max(active["max_favorable_pnl"], active["trade_pnl"])
            active["exit_time"] = row["timestamp"]
            active["exit_price"] = row["price"]
            active["exit_reason"] = row["exit_reason"]
            active["balance_after_trade"] = float(row["equity"])
            trades.append(active)
            active = None

        if changed and executed_contracts != 0:
            active = {
                "entry_time": row["timestamp"],
                "entry_price": row["price"],
                "position": int(np.sign(executed_contracts)),
                "contracts": abs(executed_contracts),
                "risk_budget": float(row["risk_budget"]),
                "trade_pnl": 0.0,
                "max_adverse_pnl": 0.0,
                "max_favorable_pnl": 0.0,
            }

        if active is not None and not (changed and previous_contracts != 0):
            active["trade_pnl"] += float(row["net_pnl"])
            active["max_adverse_pnl"] = min(active["max_adverse_pnl"], active["trade_pnl"])
            active["max_favorable_pnl"] = max(active["max_favorable_pnl"], active["trade_pnl"])

    if active is not None and not trace.empty:
        last = trace.iloc[-1]
        active["exit_time"] = last["timestamp"]
        active["exit_price"] = last["price"]
        active["exit_reason"] = "end_of_test"
        active["balance_after_trade"] = float(last["equity"])
        trades.append(active)

    ledger = pd.DataFrame(trades)
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "position",
                "contracts",
                "risk_budget",
                "trade_pnl",
                "max_adverse_pnl",
                "max_favorable_pnl",
                "exit_reason",
                "balance_after_trade",
            ]
        )
    ledger.insert(0, "trade_id", range(1, len(ledger) + 1))
    return ledger


def run_dynamic_risk_policy(
    model_name: str,
    dataset_name: str,
    market_config: MarketConfig,
    risk_config: RiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model, metadata = load_model(model_path(model_name))
    feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(metadata["feature_std"], dtype=np.float32).reshape(1, -1)
    test_df = pd.read_csv(dataset_path(dataset_name, "test"), parse_dates=["timestamp"])
    env = TradingEnv(
        test_df,
        market_config=market_config,
        random_start=False,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    close = env.close
    equity = market_config.initial_capital
    peak_equity = equity
    contracts = 0
    active_trade_pnl = 0.0
    active_risk_budget = 0.0
    cooldown_remaining = 0
    rows = []

    for step in range(len(close) - 1):
        timestamp = env.df.loc[step + 1, "timestamp"] if "timestamp" in env.df.columns else step + 1
        price = float(close[step])
        next_price = float(close[step + 1])
        position_sign = int(np.sign(contracts))
        obs = np.concatenate([env.features[step], np.array([position_sign], dtype=np.float32)]).astype(np.float32)
        raw_action = action_from_policy(model, obs, deterministic=True)
        raw_signal = int(env.action_map[raw_action])

        lookback_start = max(0, step - risk_config.volatility_lookback + 1)
        volatility_points = float(np.std(np.diff(close[lookback_start : step + 1]))) if step > lookback_start else 0.0
        drawdown = equity / max(peak_equity, 1e-12) - 1.0
        risk_mode = "normal"
        max_contracts = risk_config.max_contracts
        if drawdown <= -risk_config.kill_switch_drawdown:
            risk_mode = "kill_switch"
            max_contracts = 0
        elif drawdown <= -risk_config.protective_drawdown:
            risk_mode = "protective"
            max_contracts = 1

        proposed_contracts, risk_budget = _contracts_for_risk(
            equity, volatility_points, risk_config, market_config, max_contracts
        )
        target_contracts_before_risk = raw_signal * proposed_contracts
        target_contracts = target_contracts_before_risk
        stop_triggered = False
        take_profit_triggered = False
        exit_reason = "model_signal"

        if contracts != 0 and active_risk_budget > 0:
            if active_trade_pnl <= -risk_config.stop_loss_risk_multiple * active_risk_budget:
                stop_triggered = True
                target_contracts = 0
                exit_reason = "stop_loss"
                cooldown_remaining = risk_config.cooldown_bars
            elif active_trade_pnl >= risk_config.take_profit_risk_multiple * active_risk_budget:
                take_profit_triggered = True
                target_contracts = 0
                exit_reason = "take_profit"

        if not stop_triggered and not take_profit_triggered:
            if risk_mode == "kill_switch":
                target_contracts = 0
                exit_reason = "kill_switch"
            elif cooldown_remaining > 0 and contracts == 0:
                target_contracts = 0
                exit_reason = "cooldown"
            elif contracts != 0 and raw_signal == int(np.sign(contracts)):
                target_contracts = contracts
                exit_reason = "hold"
            elif contracts != 0 and raw_signal != int(np.sign(contracts)):
                target_contracts = 0
                exit_reason = "model_exit"
            elif target_contracts == contracts:
                exit_reason = "hold"

        previous_contracts = contracts
        changed = target_contracts != previous_contracts
        trade_size = abs(target_contracts - previous_contracts)
        cost = trade_size * _cost_per_contract(market_config)
        gross_pnl = target_contracts * (next_price - price) * market_config.contract_multiplier
        net_pnl = gross_pnl - cost
        equity += net_pnl
        peak_equity = max(peak_equity, equity)

        if changed and previous_contracts != 0:
            active_trade_pnl = 0.0
            active_risk_budget = 0.0
        if changed and target_contracts != 0:
            active_trade_pnl = 0.0
            active_risk_budget = risk_budget

        contracts = int(target_contracts)
        if contracts != 0:
            active_trade_pnl += net_pnl
        if cooldown_remaining > 0 and contracts == 0:
            cooldown_remaining -= 1

        rows.append(
            {
                "timestamp": timestamp,
                "price": next_price,
                "raw_action": raw_action,
                "raw_signal": raw_signal,
                "target_contracts_before_risk": target_contracts_before_risk,
                "previous_contracts": previous_contracts,
                "executed_contracts": contracts,
                "risk_mode": risk_mode,
                "exit_reason": exit_reason,
                "stop_triggered": stop_triggered,
                "take_profit_triggered": take_profit_triggered,
                "cooldown_remaining": cooldown_remaining,
                "volatility_points": volatility_points,
                "risk_budget": risk_budget,
                "drawdown": equity / max(peak_equity, 1e-12) - 1.0,
                "trade_size": trade_size,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "equity": equity,
            }
        )

    trace = pd.DataFrame(rows)
    ledger = _dynamic_trade_ledger(trace)
    summary = {
        "selected_model": f"{model_name}_model",
        "selected_dataset": f"{dataset_name}_test",
        "risk_method": "dynamic_balanced",
        "initial_capital": market_config.initial_capital,
        "risk_config": asdict(risk_config),
        "metrics": evaluate_trace(trace, market_config),
        "number_of_trades": int(len(ledger)),
    }
    return trace, ledger, summary


def run_risk_management(
    output_dir: Path,
    market_config: MarketConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> dict:
    cfg = market_config or MarketConfig()
    risk_cfg = risk_config or RiskConfig()
    best_path = output_dir / "best_model.json"
    if not best_path.exists():
        raise FileNotFoundError("best_model.json not found; run evaluation before risk management")

    best = json.loads(best_path.read_text(encoding="utf-8"))
    model_name = str(best["model"]).replace("_model", "")
    dataset_name = str(best["dataset"]).replace("_test", "")
    test_df = pd.read_csv(dataset_path(dataset_name, "test"), parse_dates=["timestamp"])
    static_trace = run_policy_on_dataframe(model_path(model_name), test_df, cfg)
    dynamic_trace, dynamic_ledger, dynamic_summary = run_dynamic_risk_policy(
        model_name, dataset_name, cfg, risk_cfg
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "risk_equity_curve.csv"
    trades_path = output_dir / "risk_trade_ledger.csv"
    summary_path = output_dir / "risk_summary.json"
    dynamic_equity_path = output_dir / "dynamic_risk_equity_curve.csv"
    dynamic_trades_path = output_dir / "dynamic_risk_trade_ledger.csv"
    dynamic_summary_path = output_dir / "dynamic_risk_summary.json"
    comparison_path = output_dir / "risk_comparison.csv"

    dynamic_trace.to_csv(equity_path, index=False)
    dynamic_ledger.to_csv(trades_path, index=False)
    dynamic_trace.to_csv(dynamic_equity_path, index=False)
    dynamic_ledger.to_csv(dynamic_trades_path, index=False)

    static_ledger = trade_ledger_from_trace(static_trace, cfg.initial_capital)
    static_metrics = evaluate_trace(static_trace, cfg)
    dynamic_metrics = dynamic_summary["metrics"]
    comparison = pd.DataFrame(
        [
            {
                "method": "fixed_one_contract",
                **static_metrics,
                "number_of_trades": int(len(static_ledger)),
            },
            {
                "method": "dynamic_balanced",
                **dynamic_metrics,
                "number_of_trades": int(len(dynamic_ledger)),
            },
        ]
    )
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "selected_model": best["model"],
        "selected_dataset": best["dataset"],
        "risk_method": "dynamic_balanced",
        "initial_capital": cfg.initial_capital,
        "risk_config": asdict(risk_cfg),
        "metrics": dynamic_metrics,
        "equity_curve_path": str(equity_path),
        "trade_ledger_path": str(trades_path),
        "dynamic_equity_curve_path": str(dynamic_equity_path),
        "dynamic_trade_ledger_path": str(dynamic_trades_path),
        "dynamic_summary_path": str(dynamic_summary_path),
        "comparison_path": str(comparison_path),
        "number_of_trades": int(len(dynamic_ledger)),
        "static_baseline_metrics": static_metrics,
        "static_baseline_trades": int(len(static_ledger)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dynamic_summary.update(
        {
            "equity_curve_path": str(dynamic_equity_path),
            "trade_ledger_path": str(dynamic_trades_path),
            "comparison_path": str(comparison_path),
        }
    )
    dynamic_summary_path.write_text(json.dumps(dynamic_summary, indent=2), encoding="utf-8")
    return summary
