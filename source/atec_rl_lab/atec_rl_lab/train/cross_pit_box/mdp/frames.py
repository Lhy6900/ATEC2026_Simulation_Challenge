from __future__ import annotations

import torch


def root_pos_env(env, asset) -> torch.Tensor:
    """Return root position in the replicated environment's local frame."""
    pos_w = asset.data.root_pos_w
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        return pos_w
    return pos_w - env_origins.to(device=pos_w.device, dtype=pos_w.dtype)
