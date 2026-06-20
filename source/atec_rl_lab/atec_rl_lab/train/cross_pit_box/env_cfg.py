from __future__ import annotations

import copy
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import atec_rl_lab.train.cross_pit_box.mdp as mdp
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG
from atec_rl_lab.tasks.task_d.terrain import PitAndPlatformTerrainCfg, TASK_D_TERRAIN_CFG

COUNTER650_ROBOT_POSE = (-1.75, 0.0, 0.80, 0.0)
COUNTER650_BOX_SUPPORT_POSE = (-0.85, 0.0, -0.42, 0.0)
CROSS_PIT_START_X = COUNTER650_ROBOT_POSE[0]
CROSS_PIT_DOWN_STEP_X = -1.20
CROSS_PIT_BOX_X = COUNTER650_BOX_SUPPORT_POSE[0]
CROSS_PIT_FAR_LIP_X = 0.35
CROSS_PIT_PLATFORM_X = 1.10
CROSS_PIT_SUCCESS_START_X = 1.85
CROSS_PIT_SUCCESS_FINAL_X = 2.55
CROSS_PIT_MIN_X = -2.50
CROSS_PIT_BOX_GRID_SHAPE = (24, 16)

COUNTER650_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_pitch_joint": 0.35,
}


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _build_counter650_terrain_cfg() -> TerrainImporterCfg:
    terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
    terrain_cfg.max_init_terrain_level = 0
    if terrain_cfg.terrain_generator is not None:
        terrain_cfg.terrain_generator.num_rows = 1
        terrain_cfg.terrain_generator.num_cols = 1
        terrain_cfg.terrain_generator.curriculum = False
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = (0.9, 1.0)
            pit_cfg.platform_height_range = (0.9, 1.0)
            pit_cfg.pit_depth = 1.0
    return terrain_cfg


@configclass
class CrossPitBoxSceneCfg(InteractiveSceneCfg):
    """Fixed counter-650 scene: box is already in the pit and only acts as support."""

    terrain = _build_counter650_terrain_cfg()
    robot: ArticulationCfg = MISSING
    box_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxSupport",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.38, 0.10),
                roughness=0.75,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=COUNTER650_BOX_SUPPORT_POSE[:3],
            rot=_yaw_quat(COUNTER650_BOX_SUPPORT_POSE[3]),
        ),
    )
    depth_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link",
        update_period=0.02,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(CROSS_PIT_BOX_GRID_SHAPE[0] * 0.1 - 0.1, CROSS_PIT_BOX_GRID_SHAPE[1] * 0.1 - 0.1),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=6.0,
        ray_alignment="yaw",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(1.3, 0.0, 2.5)),
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/BoxSupport",
                is_shared=True,
                track_mesh_transforms=True,
            ),
        ],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ATEC_ASSETS_MODEL_DIR}/scene/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UNITREE_G1_29DOF_DEX1_CFG.joint_names,
        scale=0.5,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-100.0, 100.0),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_29DOF_DEX1_CFG.joint_names)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_29DOF_DEX1_CFG.joint_names)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=0.1,
        )
        actions = ObsTerm(func=mdp.last_action)
        depth_heightmap = ObsTerm(
            func=mdp.raycast_heightmap,
            params={
                "sensor_cfg": SceneEntityCfg("depth_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
                "grid_shape": CROSS_PIT_BOX_GRID_SHAPE,
                "x_range": (0.0, 2.6),
                "y_range": (-1.0, 1.0),
                "z_clip": (-1.0, 1.0),
                "default_height": -1.0,
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    reset_box_support = EventTerm(
        func=mdp.reset_fixed_box_support_pose,
        mode="reset",
        params={
            "pose": COUNTER650_BOX_SUPPORT_POSE,
            "pose_noise": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.015, 0.015)},
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_counter650_state,
        mode="reset",
        params={
            "pose": COUNTER650_ROBOT_POSE,
            "pose_noise": {"x": (-0.10, 0.10), "y": (-0.08, 0.08), "yaw": (-0.10, 0.10)},
            "velocity_noise": {
                "x": (-0.05, 0.15),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.05, 0.05),
            },
            "joint_pos_overrides": COUNTER650_JOINT_POS,
            "joint_pos_noise": 0.03,
        },
    )


@configclass
class RewardsCfg:
    crossing_failure = RewTerm(
        func=mdp.crossing_failure_penalty,
        weight=-25.0,
        params={"min_height": 0.30, "max_abs_gravity_xy": 0.75, "max_abs_y": 1.25, "min_x": CROSS_PIT_MIN_X},
    )
    time_pressure = RewTerm(func=mdp.time_penalty, weight=-0.02)
    crossing_progress = RewTerm(
        func=mdp.crossing_progress,
        weight=3.0,
        params={
            "start_x": CROSS_PIT_START_X,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
        },
    )
    incremental_crossing_progress = RewTerm(
        func=mdp.incremental_crossing_progress,
        weight=10.0,
        params={
            "start_x": CROSS_PIT_START_X,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "max_step_progress": 0.02,
        },
    )
    forward_velocity = RewTerm(func=mdp.forward_velocity_reward, weight=2.0, params={"max_velocity": 1.0})
    crossing_milestones = RewTerm(
        func=mdp.crossing_milestone_reward,
        weight=8.0,
        params={
            "down_step_x": CROSS_PIT_DOWN_STEP_X,
            "box_x": CROSS_PIT_BOX_X,
            "far_lip_x": CROSS_PIT_FAR_LIP_X,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "margin": 0.15,
        },
    )
    box_support_region = RewTerm(
        func=mdp.box_support_region_reward,
        weight=0.8,
        params={
            "box_x": CROSS_PIT_BOX_X,
            "half_length": 0.45,
            "half_width": 0.55,
            "min_height": 0.35,
        },
    )
    far_platform_region = RewTerm(
        func=mdp.far_platform_region_reward,
        weight=8.0,
        params={
            "far_lip_x": CROSS_PIT_FAR_LIP_X,
            "platform_x": CROSS_PIT_PLATFORM_X,
            "half_width": 0.45,
            "min_height": 0.55,
            "margin": 0.20,
        },
    )
    crossing_success = RewTerm(
        func=mdp.crossing_success,
        weight=120.0,
        params={"success_x": CROSS_PIT_SUCCESS_START_X, "success_x_attr": "cross_pit_success_x", "min_height": 0.55},
    )
    upright = RewTerm(func=mdp.upright_reward, weight=0.3)
    lateral_deviation = RewTerm(func=mdp.lateral_deviation_l2, weight=-1.6, params={"target_y": 0.0})
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(
        func=mdp.fallen_or_tilted,
        params={"min_height": 0.30, "max_abs_gravity_xy": 0.75},
        time_out=False,
    )
    lateral_out = DoneTerm(func=mdp.lateral_out_of_bounds, params={"max_abs_y": 1.25}, time_out=False)
    backtracked = DoneTerm(func=mdp.backtracked, params={"min_x": CROSS_PIT_MIN_X}, time_out=False)
    crossed = DoneTerm(
        func=mdp.crossed_target,
        params={"success_x": CROSS_PIT_SUCCESS_START_X, "success_x_attr": "cross_pit_success_x", "min_height": 0.55},
        time_out=False,
    )


@configclass
class CurriculumCfg:
    success_x = CurrTerm(
        func=mdp.update_cross_pit_success_x,
        params={
            "start_x": CROSS_PIT_SUCCESS_START_X,
            "final_x": CROSS_PIT_SUCCESS_FINAL_X,
            "start_iteration": 1000,
            "end_iteration": 6000,
            "steps_per_iteration": 32,
        },
    )


@configclass
class CrossPitBoxG1EnvCfg(ManagerBasedRLEnvCfg):
    scene: CrossPitBoxSceneCfg = CrossPitBoxSceneCfg(num_envs=512, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=COUNTER650_ROBOT_POSE[:3],
                rot=_yaw_quat(COUNTER650_ROBOT_POSE[3]),
                joint_pos=COUNTER650_JOINT_POS,
            ),
        )
        self.scene.depth_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
