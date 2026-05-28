from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRPO_DIR = PROJECT_ROOT / "GRPO"
RAW_DATA_PATH = PROJECT_ROOT / "原始数据" / "hi1_20170701_20200609.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = GRPO_DIR / "models"
OUTPUT_DIR = GRPO_DIR / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

DATASET_NAMES = ("original", "day", "night")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class MarketConfig:
    """Trading assumptions shared by training, evaluation and risk reports."""

    contract_multiplier: float = 50.0
    transaction_cost_per_contract: float = 10.0
    slippage_points: float = 1.0
    bars_per_year: int = 252 * 330
    initial_capital: float = 5_000_000.0


@dataclass(frozen=True)
class RiskConfig:
    """Balanced dynamic risk controls for model-driven live testing."""

    risk_per_trade: float = 0.005
    max_contracts: int = 5
    protective_drawdown: float = 0.08
    kill_switch_drawdown: float = 0.12
    stop_loss_risk_multiple: float = 1.0
    take_profit_risk_multiple: float = 1.5
    cooldown_bars: int = 20
    volatility_lookback: int = 60
    min_volatility_points: float = 20.0


@dataclass(frozen=True)
class TrainConfig:
    """Default training settings for the full experiment run."""

    epochs: int = 30
    group_size: int = 8
    episode_steps: int = 2048
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    hidden_size: int = 64
    seed: int = 42


def dataset_path(dataset: str, split: str | None = None) -> Path:
    if split is None:
        return DATA_DIR / f"{dataset}_data.csv"
    return DATA_DIR / f"{dataset}_{split}.csv"


def model_path(dataset: str) -> Path:
    return MODEL_DIR / f"{dataset}_model.pt"
