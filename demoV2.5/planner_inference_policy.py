"""Inference-only loader for the box-pushing planner checkpoint."""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn


_ACTOR_WEIGHT_RE = re.compile(r"^actor\.(\d+)\.weight$")


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported planner actor activation: {name!r}")


def _extract_model_state_dict(loaded):
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


def _actor_layer_indices(state_dict: dict) -> list[int]:
    indices = []
    for key in state_dict:
        match = _ACTOR_WEIGHT_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        raise ValueError("Checkpoint does not contain actor.*.weight tensors")
    return sorted(indices)


class _ActorObsNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor | None, std: torch.Tensor | None, eps: float = 1.0e-2):
        super().__init__()
        self.eps = eps
        if mean is None or std is None:
            self.register_buffer("_mean", torch.empty(0))
            self.register_buffer("_std", torch.empty(0))
            self.enabled = False
        else:
            self.register_buffer("_mean", mean.detach().clone())
            self.register_buffer("_std", std.detach().clone())
            self.enabled = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return obs
        return (obs - self._mean) / (self._std + self.eps)


class PlannerInferencePolicy(nn.Module):
    """Minimal actor-only policy compatible with rsl_rl-style planner checkpoints."""

    def __init__(
        self,
        actor: nn.Sequential,
        *,
        actor_obs_mean: torch.Tensor | None = None,
        actor_obs_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.actor_obs_normalizer = _ActorObsNormalizer(actor_obs_mean, actor_obs_std)
        self.actor = actor

    def act_inference(self, obs):
        actor_obs = obs["policy"] if isinstance(obs, dict) else obs
        return self.actor(self.actor_obs_normalizer(actor_obs))


def build_actor_from_state_dict(state_dict: dict, *, activation: str = "elu") -> nn.Sequential:
    layers: list[nn.Module] = []
    indices = _actor_layer_indices(state_dict)
    for layer_pos, index in enumerate(indices):
        weight_key = f"actor.{index}.weight"
        bias_key = f"actor.{index}.bias"
        if bias_key not in state_dict:
            raise ValueError(f"Checkpoint is missing {bias_key}")
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        if len(weight.shape) != 2 or len(bias.shape) != 1:
            raise ValueError(f"Invalid actor layer tensor shapes for actor.{index}")
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_pos != len(indices) - 1:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


def load_planner_inference_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    expected_obs_dim: int | None = None,
    expected_action_dim: int | None = None,
    activation: str = "elu",
) -> PlannerInferencePolicy:
    try:
        loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _extract_model_state_dict(loaded)
    actor = build_actor_from_state_dict(state_dict, activation=activation)

    first_linear = next(module for module in actor if isinstance(module, nn.Linear))
    last_linear = next(module for module in reversed(actor) if isinstance(module, nn.Linear))
    if expected_obs_dim is not None and first_linear.in_features != expected_obs_dim:
        raise ValueError(
            f"Checkpoint actor expects {first_linear.in_features} obs, but environment provides {expected_obs_dim}"
        )
    if expected_action_dim is not None and last_linear.out_features != expected_action_dim:
        raise ValueError(
            f"Checkpoint actor outputs {last_linear.out_features} actions, expected {expected_action_dim}"
        )

    policy = PlannerInferencePolicy(
        actor,
        actor_obs_mean=state_dict.get("actor_obs_normalizer._mean"),
        actor_obs_std=state_dict.get("actor_obs_normalizer._std"),
    )
    policy.to(device)
    policy.eval()
    return policy
