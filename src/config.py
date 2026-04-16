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


def _validate_config(config: dict) -> None:
    schema = {
        "board_center": (list, tuple),
        "square_size": (int, float),
        "table_height": (int, float),
        "z_grasp": (int, float),
        "z_safe": (int, float),
        "n_substeps": int,
        "action_scale": (int, float),
        "goal_tolerance": (int, float),
        "placement_tolerance": (int, float),
        "max_steps_per_phase": int,
        "fetch_policy_horizon": int,
        "settle_steps": int,
        "retract_steps": int,
        "gripper_actuation_steps": int,
        "grip_contact_tolerance": (int, float),
        "piece_follow_tolerance": (int, float),
        "piece_drift_tolerance": (int, float),
        "release_settle_steps": int,
        "release_velocity_tolerance": (int, float),
        "release_gripper_open_tolerance": (int, float),
        "release_clearance_margin": (int, float),
        "final_settle_steps": int,
        "pregrasp_gripper_opening": (int, float),
        "grasp_descend_offset": (int, float),
        "close_descend_offset": (int, float),
        "close_descend_steps": int,
        "workspace_xy_margin": (int, float),
        "workspace_z_min": (int, float),
        "workspace_z_max": (int, float),
        "viewer_step_delay_sec": (int, float),
        "reachable_x_min": (int, float),
        "reachable_x_max": (int, float),
        "reachable_y_min": (int, float),
        "reachable_y_max": (int, float),
        "reachable_z_min": (int, float),
        "reachable_z_max": (int, float),
        "fetch_init_grip": (list, tuple),
        "policy_xy_radius": (int, float),
        "policy_z_min": (int, float),
        "policy_z_max": (int, float),
        "gripper_open": (int, float),
        "gripper_closed": (int, float),
        "l_finger_actuator": str,
        "r_finger_actuator": str,
        "grip_site": str,
        "white_graveyard_origin": (list, tuple),
        "black_graveyard_origin": (list, tuple),
        "graveyard_spacing": (int, float),
        "graveyard_cols": int,
        "graveyard_platform_half_extents": (list, tuple),
        "graveyard_platform_height": (int, float),
        "stockfish_paths": list,
        "stockfish_time_limit": (int, float),
        "hf_repo_id": str,
        "hf_filename": str,
        "tilt_threshold_cos": (int, float),
        "displacement_threshold": (int, float),
    }
    missing = []
    invalid_types = []
    for k, t in schema.items():
        if k not in config:
            missing.append(k)
        elif not isinstance(config[k], t):
            invalid_types.append(f"{k} (expected {t}, got {type(config[k]).__name__})")
            
    if missing or invalid_types:
        err = "Config validation failed.\n"
        if missing:
            err += f"Missing keys: {', '.join(missing)}\n"
        if invalid_types:
            err += f"Invalid types: {', '.join(invalid_types)}\n"
        raise ValueError(err)

_RUNTIME = _load_yaml_settings("runtime.yaml")
_validate_config(_RUNTIME)


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

# Fine-tuned model paths
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "checkpoints/final/tqc_robochess_final")
PRETRAINED_MODEL_PATH = HF_REPO_ID # Default to loading from hub

# Stability monitoring
TILT_THRESHOLD_COS = _RUNTIME["tilt_threshold_cos"]
DISPLACEMENT_THRESHOLD = _RUNTIME["displacement_threshold"]

# Piece names for easy lookup
PIECE_NAMES = [
    f"{color}_{p_type}{suffix}"
    for color in ("w", "b")
    for p_type, suffix in (
        ("pawn_1", ""), ("pawn_2", ""), ("pawn_3", ""), ("pawn_4", ""),
        ("pawn_5", ""), ("pawn_6", ""), ("pawn_7", ""), ("pawn_8", ""),
        ("rook_1", ""), ("rook_2", ""), ("knight_1", ""), ("knight_2", ""),
        ("bishop_1", ""), ("bishop_2", ""), ("queen", ""), ("king", "")
    )
]
# Add spares too if needed for training awareness
PIECE_NAMES += [
    f"{color}_spare_{p_type}_{count}"
    for color in ("w", "b")
    for p_type in ("queen", "rook", "bishop", "knight")
    for count in (1, 2)
]
