from __future__ import annotations

from isaaclab.utils import configclass

from .env_cfg_v2 import CrossPitBoxG1V2EnvCfg
from .env_cfg_v2 import ObservationsCfg as CrossPitBoxV2ObservationsCfg


@configclass
class BlindObservationsCfg(CrossPitBoxV2ObservationsCfg):
    @configclass
    class PolicyCfg(CrossPitBoxV2ObservationsCfg.PolicyCfg):
        depth_heightmap = None

    @configclass
    class CriticCfg(CrossPitBoxV2ObservationsCfg.CriticCfg):
        depth_heightmap = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CrossPitBoxG1V25EnvCfg(CrossPitBoxG1V2EnvCfg):
    observations: BlindObservationsCfg = BlindObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_scanner = None
