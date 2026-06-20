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

COUNTER650_ROBOT_POSE = (
    2.4567047357559204,
    -0.9189060926437378,
    0.733408510684967,
    0.9904746413230896,
    0.033361710608005524,
    -0.053021665662527084,
    0.12262003868818283,
)
COUNTER650_BOX_SUPPORT_POSE = (
    4.363436430692673,
    -0.7996934652328491,
    -0.6000000238418579,
    0.707106351852417,
    -1.4099019608693197e-05,
    0.7071070671081543,
    1.3773496903013438e-05,
)
CROSS_PIT_START_X = 2.4567047357559204
CROSS_PIT_DOWN_STEP_X = 3.72
CROSS_PIT_BOX_X = 4.363436430692673
CROSS_PIT_TARGET_Y = -0.7996934652328491
CROSS_PIT_BOX_FRONT_X = 4.0634
CROSS_PIT_BOX_BACK_X = 4.6635
CROSS_PIT_BOX_TOP_Z = -0.20
CROSS_PIT_BOX_STAGE_TARGET_X = 4.363436430692673
CROSS_PIT_NEAR_BOX_GAP_X = 0.34
CROSS_PIT_FAR_LIP_X = 4.68
CROSS_PIT_PLATFORM_X = 5.10
CROSS_PIT_SUCCESS_START_X = 3.00
CROSS_PIT_SUCCESS_FINAL_X = 7.80
CROSS_PIT_MIN_X = 1.7067047357559204
CROSS_PIT_BOX_GRID_SHAPE = (24, 16)

COUNTER650_JOINT_POS = {
    "left_hip_pitch_joint": -0.2877374589443207,
    "right_hip_pitch_joint": -0.2034483551979065,
    "waist_yaw_joint": -0.3175305426120758,
    "left_hip_roll_joint": -0.2941874563694,
    "right_hip_roll_joint": -0.12947045266628265,
    "waist_roll_joint": -0.135825514793396,
    "left_hip_yaw_joint": -0.09720142930746078,
    "right_hip_yaw_joint": 0.0031293488573282957,
    "waist_pitch_joint": 0.30181464552879333,
    "left_knee_joint": 0.7740746140480042,
    "right_knee_joint": 0.9507670402526855,
    "left_shoulder_pitch_joint": 0.33914411067962646,
    "right_shoulder_pitch_joint": 0.3461231291294098,
    "left_ankle_pitch_joint": -0.3717866539955139,
    "right_ankle_pitch_joint": -0.6566326022148132,
    "left_shoulder_roll_joint": 0.18020488321781158,
    "right_shoulder_roll_joint": -0.1690298467874527,
    "left_ankle_roll_joint": 0.1453840285539627,
    "right_ankle_roll_joint": 0.040145665407180786,
    "left_shoulder_yaw_joint": -5.641000097966753e-05,
    "right_shoulder_yaw_joint": 0.004989289212971926,
    "left_elbow_joint": 0.8753758072853088,
    "right_elbow_joint": 0.8796797394752502,
    "left_wrist_roll_joint": 2.9499602533178404e-05,
    "right_wrist_roll_joint": 3.065022247028537e-05,
    "left_wrist_pitch_joint": 0.002598497550934553,
    "right_wrist_pitch_joint": 0.0031040343455970287,
    "left_wrist_yaw_joint": -9.892250818666071e-05,
    "right_wrist_yaw_joint": 0.0006147791864350438,
    "left_hand_Joint1_1": 0.024062851443886757,
    "left_hand_Joint2_1": 0.02393471635878086,
    "right_hand_Joint1_1": 0.023989850655198097,
    "right_hand_Joint2_1": 0.024038616567850113,
}


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _pose_quat(pose: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(pose) == 7:
        return pose[3:7]
    return _yaw_quat(pose[3])


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
            rot=_pose_quat(COUNTER650_BOX_SUPPORT_POSE),
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
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_pos_env = ObsTerm(func=mdp.base_pos_env, clip=(-100.0, 100.0))
        box_center_lateral_error = ObsTerm(
            func=mdp.box_center_lateral_error,
            params={"box_y": COUNTER650_BOX_SUPPORT_POSE[1]},
            clip=(-100.0, 100.0),
        )
        abs_lateral_error = ObsTerm(
            func=mdp.abs_lateral_error,
            params={"box_y": COUNTER650_BOX_SUPPORT_POSE[1]},
            clip=(-100.0, 100.0),
        )
        target_x_error = ObsTerm(
            func=mdp.target_x_error,
            params={"success_x": CROSS_PIT_SUCCESS_START_X, "success_x_attr": "cross_pit_success_x"},
            clip=(-100.0, 100.0),
        )
        crossing_stage_privileged = ObsTerm(
            func=mdp.crossing_stage_privileged,
            params={
                "near_lip_x": CROSS_PIT_DOWN_STEP_X,
                "box_front_x": CROSS_PIT_BOX_FRONT_X,
                "box_center_x": CROSS_PIT_BOX_X,
                "box_back_x": CROSS_PIT_BOX_BACK_X,
                "far_lip_x": CROSS_PIT_FAR_LIP_X,
                "far_platform_x": CROSS_PIT_PLATFORM_X,
                "success_x": CROSS_PIT_SUCCESS_START_X,
                "success_x_attr": "cross_pit_success_x",
            },
            clip=(-100.0, 100.0),
        )
        box_gap_privileged = ObsTerm(
            func=mdp.box_gap_privileged,
            params={
                "near_lip_x": CROSS_PIT_DOWN_STEP_X,
                "box_front_x": CROSS_PIT_BOX_FRONT_X,
                "box_center_x": CROSS_PIT_BOX_X,
                "box_back_x": CROSS_PIT_BOX_BACK_X,
                "far_lip_x": CROSS_PIT_FAR_LIP_X,
                "far_platform_x": CROSS_PIT_PLATFORM_X,
                "success_x": CROSS_PIT_SUCCESS_START_X,
                "success_x_attr": "cross_pit_success_x",
            },
            clip=(-100.0, 100.0),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_29DOF_DEX1_CFG.joint_names)},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_G1_29DOF_DEX1_CFG.joint_names)},
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
            "pose_noise": {},
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_counter650_state,
        mode="reset",
        params={
            "pose": COUNTER650_ROBOT_POSE,
            "pose_noise": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.2617993877991494, 0.2617993877991494)},
            "velocity_noise": {
                "x": (-0.10, 0.10),
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
        params={
            "min_height": 0.30,
            "max_abs_gravity_xy": 0.75,
            "max_abs_y": 1.25,
            "min_x": CROSS_PIT_MIN_X,
            "target_y": CROSS_PIT_TARGET_Y,
        },
    )
    time_pressure = RewTerm(func=mdp.time_penalty, weight=-0.02)
    stage_time_pressure = RewTerm(
        func=mdp.stage_time_penalty,
        weight=-0.08,
        params={
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_BOX_FRONT_X,
        },
    )
    crossing_progress = None
    pbrs_course_progress = RewTerm(
        func=mdp.potential_based_x_progress,
        weight=20.0,
        params={
            "start_x": CROSS_PIT_START_X,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "gamma": 0.99,
            "potential_scale": 50.0,
            "target_y": CROSS_PIT_TARGET_Y,
            "safe_abs_y": 0.60,
            "min_upright": 0.60,
            "cache_key": "_cross_pit_pbrs_course_phi",
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
        weight=1.0,
        params={
            "box_x": CROSS_PIT_BOX_X,
            "box_y": CROSS_PIT_TARGET_Y,
            "half_length": 0.32,
            "half_width": 0.55,
            "min_height": 0.20,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_BOX_FRONT_X,
        },
    )
    box_gap_step = RewTerm(
        func=mdp.box_gap_step_reward,
        weight=5.0,
        params={
            "near_lip_x": CROSS_PIT_DOWN_STEP_X,
            "box_front_x": CROSS_PIT_BOX_FRONT_X,
            "box_center_x": CROSS_PIT_BOX_X,
            "target_y": CROSS_PIT_TARGET_Y,
            "min_forward_velocity": 0.55,
            "max_abs_y": 0.35,
            "max_up_velocity": 0.35,
            "margin": 0.12,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_DOWN_STEP_X,
        },
    )
    box_stability = RewTerm(
        func=mdp.box_stability_reward,
        weight=0.5,
        params={
            "box_x": CROSS_PIT_BOX_X,
            "box_y": CROSS_PIT_TARGET_Y,
            "half_length": 0.32,
            "half_width": 0.45,
            "min_height": 0.20,
            "min_forward_velocity": 0.15,
            "max_forward_velocity": 0.65,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_BOX_FRONT_X,
        },
    )
    stage_progress_to_target = None
    pbrs_stage_progress = RewTerm(
        func=mdp.potential_based_x_progress,
        weight=15.0,
        params={
            "start_x": CROSS_PIT_BOX_FRONT_X,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "gamma": 0.99,
            "potential_scale": 50.0,
            "target_y": CROSS_PIT_TARGET_Y,
            "safe_abs_y": 0.45,
            "min_upright": 0.60,
            "unlock_success_x": CROSS_PIT_BOX_STAGE_TARGET_X,
            "cache_key": "_cross_pit_pbrs_stage_phi",
        },
    )
    stage_stall = RewTerm(
        func=mdp.stage_stall_penalty,
        weight=-5.0,
        params={
            "stage_start_x": CROSS_PIT_BOX_FRONT_X,
            "target_y": CROSS_PIT_TARGET_Y,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_BOX_FRONT_X,
            "target_margin": 0.20,
            "min_forward_velocity": 0.12,
            "safe_abs_y": 0.45,
            "min_upright": 0.75,
        },
    )
    box_parking = RewTerm(
        func=mdp.box_parking_penalty,
        weight=-6.0,
        params={
            "box_front_x": CROSS_PIT_BOX_FRONT_X,
            "target_y": CROSS_PIT_TARGET_Y,
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "unlock_success_x": CROSS_PIT_BOX_FRONT_X,
            "target_margin": 0.12,
            "max_forward_velocity": 0.08,
            "max_abs_y": 0.35,
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
            "target_y": CROSS_PIT_TARGET_Y,
        },
    )
    far_side_commit = RewTerm(
        func=mdp.far_side_commit_reward,
        weight=6.0,
        params={
            "far_lip_x": CROSS_PIT_FAR_LIP_X,
            "platform_x": CROSS_PIT_PLATFORM_X,
            "target_y": CROSS_PIT_TARGET_Y,
            "half_width": 0.45,
            "min_height": 0.50,
            "min_forward_velocity": 0.20,
            "margin": 0.20,
        },
    )
    crossing_success = RewTerm(
        func=mdp.crossing_success,
        weight=120.0,
        params={
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "min_height": 0.55,
            "max_abs_y": 0.55,
            "target_y": CROSS_PIT_TARGET_Y,
        },
    )
    upright = RewTerm(func=mdp.upright_reward, weight=0.3)
    lateral_deviation = RewTerm(func=mdp.lateral_deviation_l2, weight=-1.6, params={"target_y": CROSS_PIT_TARGET_Y})
    lateral_corridor = RewTerm(
        func=mdp.lateral_corridor_barrier,
        weight=-3.0,
        params={"safe_abs_y": 0.45, "max_abs_y": 1.25, "target_y": CROSS_PIT_TARGET_Y},
    )
    lateral_velocity_l2 = RewTerm(func=mdp.lateral_velocity_l2, weight=-0.2)
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
    lateral_out = DoneTerm(func=mdp.lateral_out_of_bounds, params={"max_abs_y": 1.25, "target_y": CROSS_PIT_TARGET_Y}, time_out=False)
    backtracked = DoneTerm(func=mdp.backtracked, params={"min_x": CROSS_PIT_MIN_X}, time_out=False)
    crossed = DoneTerm(
        func=mdp.crossed_target,
        params={
            "success_x": CROSS_PIT_SUCCESS_START_X,
            "success_x_attr": "cross_pit_success_x",
            "min_height": 0.55,
            "max_abs_y": 0.55,
            "target_y": CROSS_PIT_TARGET_Y,
        },
        time_out=False,
    )


@configclass
class CurriculumCfg:
    success_x = CurrTerm(
        func=mdp.update_cross_pit_success_x_sticky_stage,
        params={
            "start_x": CROSS_PIT_SUCCESS_START_X,
            "final_x": CROSS_PIT_SUCCESS_FINAL_X,
            "sticky_start_x": CROSS_PIT_BOX_FRONT_X,
            "retreat_floor_x": CROSS_PIT_BOX_STAGE_TARGET_X,
            "steps_per_iteration": 32,
            "min_iteration": 0,
            "update_interval_iterations": 10,
            "advance_step": 0.05,
            "fast_advance_step": 0.10,
            "retreat_step": 0.025,
            "success_threshold": 0.75,
            "fast_success_threshold": 0.90,
            "max_fallen_rate": 0.25,
            "max_lateral_out_rate": 0.15,
            "retreat_crossed_threshold": 0.35,
            "retreat_failure_rate": 0.45,
            "timeout_plateau_advance_step": 0.20,
            "timeout_plateau_threshold": 0.85,
            "timeout_plateau_max_crossed_rate": 0.05,
            "timeout_plateau_max_fallen_rate": 0.10,
            "timeout_plateau_unlock_x": CROSS_PIT_BOX_STAGE_TARGET_X,
            "ema_alpha": 0.2,
        },
    )


@configclass
class CrossPitBoxG1V1EnvCfg(ManagerBasedRLEnvCfg):
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
                rot=_pose_quat(COUNTER650_ROBOT_POSE),
                joint_pos=COUNTER650_JOINT_POS,
            ),
        )
        self.scene.depth_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
