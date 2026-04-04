"""YAML-backed runtime configuration for RoboChess."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = str(PACKAGE_DIR / "assets")
SETTINGS_DIR = PACKAGE_DIR / "settings"
SCENE_XML = str(Path(ASSETS_DIR) / "chess_world.xml")


def _load_yaml_settings(name: str) -> dict:
    path = SETTINGS_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


_RUNTIME = _load_yaml_settings("runtime.yaml")


# Board geometry
BOARD_CENTER = _RUNTIME["board_center"]
SQUARE_SIZE = _RUNTIME["square_size"]
TABLE_HEIGHT = _RUNTIME["table_height"]
Z_GRASP = _RUNTIME["z_grasp"]
Z_SAFE = _RUNTIME["z_safe"]

# RL control
N_SUBSTEPS = _RUNTIME["n_substeps"]
ACTION_SCALE = _RUNTIME["action_scale"]
GOAL_TOLERANCE = _RUNTIME["goal_tolerance"]
PLACEMENT_TOLERANCE = _RUNTIME["placement_tolerance"]
MAX_STEPS_PER_PHASE = _RUNTIME["max_steps_per_phase"]
FETCH_POLICY_HORIZON = _RUNTIME["fetch_policy_horizon"]
SETTLE_STEPS = _RUNTIME["settle_steps"]
RETRACT_STEPS = _RUNTIME["retract_steps"]
GRIPPER_ACTUATION_STEPS = _RUNTIME["gripper_actuation_steps"]
GRIP_CONTACT_TOLERANCE = _RUNTIME["grip_contact_tolerance"]
PIECE_FOLLOW_TOLERANCE = _RUNTIME["piece_follow_tolerance"]
PIECE_DRIFT_TOLERANCE = _RUNTIME["piece_drift_tolerance"]
RELEASE_SETTLE_STEPS = _RUNTIME["release_settle_steps"]
RELEASE_VELOCITY_TOLERANCE = _RUNTIME["release_velocity_tolerance"]
RELEASE_GRIPPER_OPEN_TOLERANCE = _RUNTIME["release_gripper_open_tolerance"]
RELEASE_CLEARANCE_MARGIN = _RUNTIME["release_clearance_margin"]
FINAL_SETTLE_STEPS = _RUNTIME["final_settle_steps"]
PREGRASP_GRIPPER_OPENING = _RUNTIME["pregrasp_gripper_opening"]
GRASP_DESCEND_OFFSET = _RUNTIME["grasp_descend_offset"]
CLOSE_DESCEND_OFFSET = _RUNTIME["close_descend_offset"]
CLOSE_DESCEND_STEPS = _RUNTIME["close_descend_steps"]
WORKSPACE_XY_MARGIN = _RUNTIME["workspace_xy_margin"]
WORKSPACE_Z_MIN = _RUNTIME["workspace_z_min"]
WORKSPACE_Z_MAX = _RUNTIME["workspace_z_max"]
VIEWER_STEP_DELAY_SEC = _RUNTIME["viewer_step_delay_sec"]
REACHABLE_X_MIN = _RUNTIME["reachable_x_min"]
REACHABLE_X_MAX = _RUNTIME["reachable_x_max"]
REACHABLE_Y_MIN = _RUNTIME["reachable_y_min"]
REACHABLE_Y_MAX = _RUNTIME["reachable_y_max"]
REACHABLE_Z_MIN = _RUNTIME["reachable_z_min"]
REACHABLE_Z_MAX = _RUNTIME["reachable_z_max"]
FETCH_INIT_GRIP = _RUNTIME["fetch_init_grip"]
POLICY_XY_RADIUS = _RUNTIME["policy_xy_radius"]
POLICY_Z_MIN = _RUNTIME["policy_z_min"]
POLICY_Z_MAX = _RUNTIME["policy_z_max"]

# Gripper
GRIPPER_OPEN = _RUNTIME["gripper_open"]
GRIPPER_CLOSED = _RUNTIME["gripper_closed"]

# Actuator names
L_FINGER_ACTUATOR = _RUNTIME["l_finger_actuator"]
R_FINGER_ACTUATOR = _RUNTIME["r_finger_actuator"]
GRIP_SITE = _RUNTIME["grip_site"]

# Graveyard
WHITE_GRAVEYARD_ORIGIN = _RUNTIME["white_graveyard_origin"]
BLACK_GRAVEYARD_ORIGIN = _RUNTIME["black_graveyard_origin"]
GRAVEYARD_SPACING = _RUNTIME["graveyard_spacing"]
GRAVEYARD_COLS = _RUNTIME["graveyard_cols"]
GRAVEYARD_PLATFORM_HALF_EXTENTS = _RUNTIME["graveyard_platform_half_extents"]
GRAVEYARD_PLATFORM_HEIGHT = _RUNTIME["graveyard_platform_height"]

# Stockfish
STOCKFISH_PATHS = _RUNTIME["stockfish_paths"]
STOCKFISH_TIME_LIMIT = _RUNTIME["stockfish_time_limit"]

# Pretrained model
HF_REPO_ID = _RUNTIME["hf_repo_id"]
HF_FILENAME = _RUNTIME["hf_filename"]

# Stability monitoring
TILT_THRESHOLD_COS = _RUNTIME["tilt_threshold_cos"]
DISPLACEMENT_THRESHOLD = _RUNTIME["displacement_threshold"]
