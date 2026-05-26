from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import MarketConfig
from .features import build_feature_matrix

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # The scripts still work without inheriting from gymnasium.
    gym = None
    spaces = None


class TradingEnv(gym.Env if gym else object):
    """Single-contract long/flat/short environment for minute HI futures data."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        market_config: MarketConfig | None = None,
        max_episode_steps: int | None = None,
        random_start: bool = False,
        seed: int = 42,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        if len(df) < 3:
            raise ValueError("trading environment requires at least 3 rows")

        self.df = df.reset_index(drop=True)
        self.features, self.close, self.feature_mean, self.feature_std = build_feature_matrix(
            self.df, mean=feature_mean, std=feature_std
        )
        self.market_config = market_config or MarketConfig()
        self.max_episode_steps = max_episode_steps or (len(self.df) - 2)
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)

        self.action_map = np.array([-1, 0, 1], dtype=np.int8)
        self.current_step = 0
        self.start_step = 0
        self.end_step = len(self.df) - 2
        self.position = 0
        self.equity = self.market_config.initial_capital
        self.trace: list[dict[str, Any]] = []

        obs_size = self.features.shape[1] + 1
        if spaces:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self.features[self.current_step], np.array([self.position], dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = max(0, len(self.df) - self.max_episode_steps - 2)
        self.start_step = int(self.rng.integers(0, max_start + 1)) if self.random_start and max_start > 0 else 0
        self.current_step = self.start_step
        self.end_step = min(len(self.df) - 2, self.start_step + self.max_episode_steps)
        self.position = 0
        self.equity = self.market_config.initial_capital
        self.trace = []
        return self._observation(), {"market_config": asdict(self.market_config)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action < 0 or action >= len(self.action_map):
            raise ValueError(f"invalid action {action}; expected 0, 1 or 2")

        target_position = int(self.action_map[action])
        previous_position = self.position
        trade_size = abs(target_position - previous_position)
        price = float(self.close[self.current_step])
        next_price = float(self.close[self.current_step + 1])

        cost = trade_size * (
            self.market_config.transaction_cost_per_contract
            + self.market_config.slippage_points * self.market_config.contract_multiplier
        )
        gross_pnl = target_position * (next_price - price) * self.market_config.contract_multiplier
        net_pnl = gross_pnl - cost
        self.equity += net_pnl
        self.position = target_position

        info = {
            "timestamp": self.df.loc[self.current_step + 1, "timestamp"]
            if "timestamp" in self.df.columns
            else self.current_step + 1,
            "price": next_price,
            "action": action,
            "previous_position": previous_position,
            "position": target_position,
            "trade_size": trade_size,
            "gross_pnl": gross_pnl,
            "cost": cost,
            "net_pnl": net_pnl,
            "equity": self.equity,
        }
        self.trace.append(info)

        self.current_step += 1
        terminated = self.current_step >= self.end_step
        reward = net_pnl / max(self.market_config.initial_capital, 1.0)
        return self._observation(), float(reward), terminated, False, info

    def trace_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trace)
