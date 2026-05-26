from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATASET_NAMES, OUTPUT_DIR, MarketConfig, TrainConfig, dataset_path
from src.grpo import train_grpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO-style policies for HI futures datasets.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES), choices=list(DATASET_NAMES))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--group-size", type=int, default=TrainConfig.group_size)
    parser.add_argument("--episode-steps", type=int, default=TrainConfig.episode_steps)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(
        epochs=args.epochs,
        group_size=args.group_size,
        episode_steps=args.episode_steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    market_config = MarketConfig()

    for dataset in args.datasets:
        train_df = pd.read_csv(dataset_path(dataset, "train"), parse_dates=["timestamp"])
        validation_df = pd.read_csv(dataset_path(dataset, "validation"), parse_dates=["timestamp"])
        result = train_grpo(dataset, train_df, validation_df, OUTPUT_DIR, train_config, market_config)
        print(f"trained {dataset}: {result['model_path']}")


if __name__ == "__main__":
    main()
