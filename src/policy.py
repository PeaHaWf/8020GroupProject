from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Imported lazily by scripts so data preparation does not need torch.
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for model training/evaluation. Install it with `pip install torch`.")


class PolicyNetwork(nn.Module if nn else object):
    def __init__(self, input_dim: int, hidden_size: int = 64, action_dim: int = 3) -> None:
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def action_from_policy(model: PolicyNetwork, observation: np.ndarray, deterministic: bool = True) -> int:
    require_torch()
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            return int(torch.argmax(logits).item())
        return int(dist.sample().item())


def save_model(path: Path, model: PolicyNetwork, metadata: dict) -> None:
    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(path: Path) -> tuple[PolicyNetwork, dict]:
    require_torch()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    model = PolicyNetwork(
        input_dim=int(metadata["input_dim"]),
        hidden_size=int(metadata.get("hidden_size", 64)),
        action_dim=int(metadata.get("action_dim", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, metadata
