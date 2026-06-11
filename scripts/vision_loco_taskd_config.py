"""Default planner-to-vision-locomotion handoff config for TaskD playback."""

DEFAULT_VISION_LOCO_TASKD_CONFIG = {
    "startup_motion": {
        "enabled": False,
        "stop_on_done": False,
        "back_steps": 100,
        "left_steps": 500,
        "forward_steps": 0,
        "back_command": [-0.2, 0.0, 0.0],
        "left_command": [0.2, 0.5, -0.15],
        "forward_command": [0.4, 0.0, 0.1],
    },
    "switch": {
        "start_with_vision_loco": False,
        "start_with_vision_loco_settle_steps": 0,
        "box_in_pit_hold_steps": 1,
        "settle_steps_after_switch": 50,
        "settle_command": [-0.0, 0.2, 0.0],
    },
    "loco": {
        "policy_path": "./demo/fsc_visual_policy.pt",
        "metadata_path": "./demo/sim2sim_metadata.json",
        "forward_velocity": 0.9,
        "yaw_kp": 0.0,
    },
    "action_compensation": {
        "enabled": True,
        "blend": 1.0,
        "max_extra_target_delta": 0.5,
    },
    "ditch": {
        "width": None,
        "length": 7.0,
        "depth": 1.0,
        "yaw_deg": 90.0,
        "center_xy": None,
    },
    "height_map": {
        "no_print": False,
        "print_interval": 2,
        "print_window": 11,
        "force_ground_behind": {
            "enabled": False,
            "x_max": 0.1,
            "y_min": -0.4,
            "y_max": 0.4,
        },
        "after_gap": {
            "enabled": False,
            "margin": 0.0,
            "vel_command": [0.8, 0.0, 0.22],
        },
        "in_gap_raise_forward_ground": {
            "enabled": False,
            "x_min": 0.0,
            "height": 0.12,
        },
        "box_fill": {
            "enabled": True,
            "size_x": 0.8,
            "size_y": 1.0,
            "height": 0.6,
            "margin": 0.05,
        },
    },
    "success": {
        "cross_x_threshold": 3.5,
    },
}
