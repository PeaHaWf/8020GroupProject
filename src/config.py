from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "原始数据" / "hi1_20170701_20200609.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
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
class TrainConfig:
    """Small defaults keep the pipeline runnable on a laptop."""

    epochs: int = 8
    group_size: int = 4
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
