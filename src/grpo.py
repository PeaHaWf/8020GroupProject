from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketConfig, TrainConfig, model_path
from .metrics import evaluate_trace
from .policy import PolicyNetwork, require_torch, save_model, torch
from .trading_env import TradingEnv


def set_seed(seed: int) -> None:
    require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_episode(model: PolicyNetwork, env: TradingEnv) -> dict:
    obs, _ = env.reset()
    log_probs = []
    entropies = []
    rewards = []
    done = False

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = model(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        rewards.append(float(reward))
        done = terminated or truncated

    return {
        "return": float(np.sum(rewards)),
        "log_prob_sum": torch.stack(log_probs).sum(),
        "entropy_mean": torch.stack(entropies).mean(),
        "trace": env.trace_frame(),
    }


def validate_policy(
    model: PolicyNetwork,
    df: pd.DataFrame,
    market_config: MarketConfig,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> dict[str, float]:
    from .policy import action_from_policy

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
    return evaluate_trace(env.trace_frame(), market_config)


def train_grpo(
    dataset_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    train_config: TrainConfig | None = None,
    market_config: MarketConfig | None = None,
) -> dict:
    require_torch()
    cfg = train_config or TrainConfig()
    market_cfg = market_config or MarketConfig()
    set_seed(cfg.seed)

    probe_env = TradingEnv(train_df, market_config=market_cfg, max_episode_steps=cfg.episode_steps, random_start=True)
    obs, _ = probe_env.reset()
    feature_mean = probe_env.feature_mean
    feature_std = probe_env.feature_std
    model = PolicyNetwork(input_dim=len(obs), hidden_size=cfg.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    rows = []
    for epoch in range(1, cfg.epochs + 1):
        episodes = []
        for offset in range(cfg.group_size):
            env = TradingEnv(
                train_df,
                market_config=market_cfg,
                max_episode_steps=cfg.episode_steps,
                random_start=True,
                seed=cfg.seed + epoch * 100 + offset,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            episodes.append(_sample_episode(model, env))

        returns = np.array([episode["return"] for episode in episodes], dtype=np.float32)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_terms = [
            -float(advantage) * episode["log_prob_sum"] - cfg.entropy_coef * episode["entropy_mean"]
            for advantage, episode in zip(advantages, episodes, strict=True)
        ]
        loss = torch.stack(policy_terms).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        validation_metrics = validate_policy(model, validation_df, market_cfg, feature_mean, feature_std)
        rows.append(
            {
                "dataset": dataset_name,
                "epoch": epoch,
                "loss": float(loss.detach().item()),
                "mean_group_return": float(returns.mean()),
                "std_group_return": float(returns.std()),
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{dataset_name}_training_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)

    metadata = {
        "dataset": dataset_name,
        "input_dim": len(obs),
        "hidden_size": cfg.hidden_size,
        "action_dim": 3,
        "feature_mean": feature_mean.squeeze(0).tolist(),
        "feature_std": feature_std.squeeze(0).tolist(),
        "train_config": asdict(cfg),
        "market_config": asdict(market_cfg),
    }
    save_model(model_path(dataset_name), model, metadata)
    (output_dir / f"{dataset_name}_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"model_path": str(model_path(dataset_name)), "log_path": str(log_path), "metadata": metadata}
