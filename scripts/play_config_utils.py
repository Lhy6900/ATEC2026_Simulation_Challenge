"""Small configuration helpers for local play/debug scripts."""

import torch


def _set_none_if_present(obj, name: str) -> bool:
    if obj is None or not hasattr(obj, name):
        return False
    setattr(obj, name, None)
    return True


def disable_camera_observations(env_cfg) -> list[str]:
    """Disable camera sensors and image observation terms on an IsaacLab env cfg."""
    disabled: list[str] = []

    scene = getattr(env_cfg, "scene", None)
    for name in ("head_camera", "ee_camera", "ee_dual_camera"):
        if _set_none_if_present(scene, name):
            disabled.append(f"scene.{name}")

    observations = getattr(env_cfg, "observations", None)
    image_obs = getattr(observations, "image", None)
    for name in (
        "head_rgb",
        "head_depth",
        "ee_rgb",
        "ee_depth",
        "ee_dual_rgb",
        "ee_dual_depth",
    ):
        if _set_none_if_present(image_obs, name):
            disabled.append(f"observations.image.{name}")

    return disabled


def make_zero_actions(env, num_envs: int, device: str, fallback_width: int | None = None) -> torch.Tensor:
    action_width = fallback_width
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
        action_width = int(shape[-1])
    if action_width is None:
        raise ValueError("Cannot infer action width for zero-action debug run.")
    return torch.zeros((int(num_envs), int(action_width)), dtype=torch.float32, device=device)


def tensor_mean_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(value)


def tensor_any_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.bool().any().item())
    return bool(value)
