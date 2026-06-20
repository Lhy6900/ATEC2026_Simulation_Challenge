from __future__ import annotations

from isaaclab.utils import configclass

from .env_cfg_v1 import CrossPitBoxG1V1EnvCfg
from .env_cfg_v1 import ObservationsCfg as CrossPitBoxV1ObservationsCfg


@configclass
class BlindObservationsCfg(CrossPitBoxV1ObservationsCfg):
    @configclass
    class PolicyCfg(CrossPitBoxV1ObservationsCfg.PolicyCfg):
        depth_heightmap = None

    @configclass
    class CriticCfg(CrossPitBoxV1ObservationsCfg.CriticCfg):
        depth_heightmap = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CrossPitBoxG1V15EnvCfg(CrossPitBoxG1V1EnvCfg):
    observations: BlindObservationsCfg = BlindObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_scanner = None
