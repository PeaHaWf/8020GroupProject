from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketConfig, dataset_path, model_path
from .metrics import evaluate_trace
from .policy import action_from_policy, load_model
from .trading_env import TradingEnv


EVALUATION_COMBOS = [
    ("original", "original"),
    ("original", "day"),
    ("original", "night"),
    ("day", "day"),
    ("night", "night"),
]


def run_policy_on_dataframe(model_path_value: Path, df: pd.DataFrame, market_config: MarketConfig) -> pd.DataFrame:
    model, metadata = load_model(model_path_value)
    feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(metadata["feature_std"], dtype=np.float32).reshape(1, -1)
    env = TradingEnv(
        df,
        market_config=market_config,
        random_start=False,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action = action_from_policy(model, obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.trace_frame()


def evaluate_combinations(output_dir: Path, market_config: MarketConfig | None = None) -> pd.DataFrame:
    cfg = market_config or MarketConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, dataset_name in EVALUATION_COMBOS:
        test_df = pd.read_csv(dataset_path(dataset_name, "test"), parse_dates=["timestamp"])
        trace = run_policy_on_dataframe(model_path(model_name), test_df, cfg)
        trace_path = output_dir / f"trace_{model_name}_model_on_{dataset_name}_test.csv"
        trace.to_csv(trace_path, index=False)

        rows.append(
            {
                "model": f"{model_name}_model",
                "dataset": f"{dataset_name}_test",
                **evaluate_trace(trace, cfg),
                "trace_path": str(trace_path),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "evaluation_results.csv", index=False)

    ranked = results.sort_values(
        by=["log_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    (output_dir / "best_model.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return results
