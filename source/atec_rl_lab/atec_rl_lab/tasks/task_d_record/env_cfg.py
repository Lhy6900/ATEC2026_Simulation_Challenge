from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import atec_rl_lab.tasks.task_d_record.mdp as record_mdp
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvG1Cfg


@configclass
class VisionHeightScanCfg(ObsGroup):
    height_map = ObsTerm(
        func=record_mdp.vision_elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TaskDRecordEnvG1Cfg(TaskDEnvG1Cfg):
    """TaskD-G1 with privileged vision height map, only for offline recording."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            update_period=self.decimation * self.sim.dt,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.vision_height_scan = VisionHeightScanCfg()
