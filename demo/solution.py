from typing import Any

_policy_name = None
_impl = None
_lidar_perception = None
_lidar_perception_key = None


def _ensure_impl():
    global _impl
    if _impl is not None:
        return
    if _policy_name == "blind":
        try:
            from solution_blind import BlindLocoSolution
        except ImportError:
            from demo.solution_blind import BlindLocoSolution
        _impl = BlindLocoSolution()
    else:
        try:
            from solution_homie import HomieSolution
        except ImportError:
            from demo.solution_homie import HomieSolution
        _impl = HomieSolution()


def set_policy(name: str) -> bool:
    global _policy_name, _impl
    name = "blind" if name == "blind" else "gr00t"
    if name == _policy_name:
        return False
    _policy_name = name
    _impl = None
    return True


def get_policy_name() -> str:
    return _policy_name or "gr00t"


def get_lidar_perception(num_envs: int, device: str = "cuda", prior=None, init_samples: int = 5):
    global _lidar_perception, _lidar_perception_key
    key = (int(num_envs), str(device), int(init_samples), id(prior))
    if _lidar_perception is not None and _lidar_perception_key == key:
        return _lidar_perception
    try:
        from demo.lidar_perception import TaskDLidarGpuPerception
    except ImportError:
        from lidar_perception import TaskDLidarGpuPerception
    _lidar_perception = TaskDLidarGpuPerception(
        num_envs=int(num_envs),
        device=device,
        prior=prior,
        init_samples=int(init_samples),
    )
    _lidar_perception_key = key
    return _lidar_perception


class AlgSolution:
    """Platform entrypoint that dispatches to the selected policy."""

    def __init__(self):
        _ensure_impl()

    def reset(self, **kwargs):
        return _impl.reset(**kwargs)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        if hasattr(_impl, "get_action_spec"):
            return _impl.get_action_spec()
        return {}

    def predicts(self, obs, current_score):
        return _impl.predicts(obs, current_score)

    def set_keyboard_command(self, nav_cmd=None, height_cmd=None):
        if hasattr(_impl, "set_keyboard_command"):
            _impl.set_keyboard_command(nav_cmd, height_cmd)

    def get_debug_snapshot(self):
        if hasattr(_impl, "get_debug_snapshot"):
            return _impl.get_debug_snapshot()
        return {}
