"""
YAML-backed runtime configuration for RoboChess.

This module provides a centralized configuration manager for RoboChess,
handling YAML loading, validation, and access to runtime settings.
"""

import os
from pathlib import Path
import yaml


class ConfigManager:
    """Manages application configuration loaded from YAML settings."""

    def __init__(self, package_dir=None):
        """
        Initialize the configuration manager.
        
        Args:
            package_dir: Base directory for the package (defaults to parent of this file).
        """
        if package_dir is None:
            self.package_dir = Path(__file__).resolve().parent
        else:
            self.package_dir = Path(package_dir)
            
        self.assets_dir = str(self.package_dir / "assets")
        self.settings_dir = self.package_dir / "settings"
        self.scene_xml = str(Path(self.assets_dir) / "chess_world.xml")
        
        self.runtime_settings = self._load_yaml_settings("runtime.yaml")
        self._validate_config(self.runtime_settings)

    def _load_yaml_settings(self, name):
        """
        Load settings from a YAML file.
        
        Args:
            name: Name of the YAML file.
            
        Returns:
            Dictionary containing the loaded settings.
        """
        path = self.settings_dir / name
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _validate_config(self, config):
        """
        Validate the loaded configuration against a schema.
        
        Args:
            config: The configuration dictionary to validate.
            
        Raises:
            ValueError: If validation fails.
        """
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
            "local_model_path": str,
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

    def get(self, key):
        """
        Retrieve a configuration value by key.
        
        Args:
            key: The configuration key to lookup.
            
        Returns:
            The value associated with the key.
        """
        return self.runtime_settings.get(key)


# Initialize global config manager
_MANAGER = ConfigManager()

# Export directories and files
PACKAGE_DIR = _MANAGER.package_dir
ASSETS_DIR = _MANAGER.assets_dir
SETTINGS_DIR = _MANAGER.settings_dir
SCENE_XML = _MANAGER.scene_xml

# Export individual configuration constants
BOARD_CENTER = _MANAGER.get("board_center")
SQUARE_SIZE = _MANAGER.get("square_size")
TABLE_HEIGHT = _MANAGER.get("table_height")
Z_GRASP = _MANAGER.get("z_grasp")
Z_SAFE = _MANAGER.get("z_safe")
N_SUBSTEPS = _MANAGER.get("n_substeps")
ACTION_SCALE = _MANAGER.get("action_scale")
GOAL_TOLERANCE = _MANAGER.get("goal_tolerance")
PLACEMENT_TOLERANCE = _MANAGER.get("placement_tolerance")
MAX_STEPS_PER_PHASE = _MANAGER.get("max_steps_per_phase")
FETCH_POLICY_HORIZON = _MANAGER.get("fetch_policy_horizon")
SETTLE_STEPS = _MANAGER.get("settle_steps")
RETRACT_STEPS = _MANAGER.get("retract_steps")
GRIPPER_ACTUATION_STEPS = _MANAGER.get("gripper_actuation_steps")
GRIP_CONTACT_TOLERANCE = _MANAGER.get("grip_contact_tolerance")
PIECE_FOLLOW_TOLERANCE = _MANAGER.get("piece_follow_tolerance")
PIECE_DRIFT_TOLERANCE = _MANAGER.get("piece_drift_tolerance")
RELEASE_SETTLE_STEPS = _MANAGER.get("release_settle_steps")
RELEASE_VELOCITY_TOLERANCE = _MANAGER.get("release_velocity_tolerance")
RELEASE_GRIPPER_OPEN_TOLERANCE = _MANAGER.get("release_gripper_open_tolerance")
RELEASE_CLEARANCE_MARGIN = _MANAGER.get("release_clearance_margin")
FINAL_SETTLE_STEPS = _MANAGER.get("final_settle_steps")
PREGRASP_GRIPPER_OPENING = _MANAGER.get("pregrasp_gripper_opening")
GRASP_DESCEND_OFFSET = _MANAGER.get("grasp_descend_offset")
CLOSE_DESCEND_OFFSET = _MANAGER.get("close_descend_offset")
CLOSE_DESCEND_STEPS = _MANAGER.get("close_descend_steps")
WORKSPACE_XY_MARGIN = _MANAGER.get("workspace_xy_margin")
WORKSPACE_Z_MIN = _MANAGER.get("workspace_z_min")
WORKSPACE_Z_MAX = _MANAGER.get("workspace_z_max")
VIEWER_STEP_DELAY_SEC = _MANAGER.get("viewer_step_delay_sec")
REACHABLE_X_MIN = _MANAGER.get("reachable_x_min")
REACHABLE_X_MAX = _MANAGER.get("reachable_x_max")
REACHABLE_Y_MIN = _MANAGER.get("reachable_y_min")
REACHABLE_Y_MAX = _MANAGER.get("reachable_y_max")
REACHABLE_Z_MIN = _MANAGER.get("reachable_z_min")
REACHABLE_Z_MAX = _MANAGER.get("reachable_z_max")
FETCH_INIT_GRIP = _MANAGER.get("fetch_init_grip")
POLICY_XY_RADIUS = _MANAGER.get("policy_xy_radius")
POLICY_Z_MIN = _MANAGER.get("policy_z_min")
POLICY_Z_MAX = _MANAGER.get("policy_z_max")
GRIPPER_OPEN = _MANAGER.get("gripper_open")
GRIPPER_CLOSED = _MANAGER.get("gripper_closed")
L_FINGER_ACTUATOR = _MANAGER.get("l_finger_actuator")
R_FINGER_ACTUATOR = _MANAGER.get("r_finger_actuator")
GRIP_SITE = _MANAGER.get("grip_site")
WHITE_GRAVEYARD_ORIGIN = _MANAGER.get("white_graveyard_origin")
BLACK_GRAVEYARD_ORIGIN = _MANAGER.get("black_graveyard_origin")
GRAVEYARD_SPACING = _MANAGER.get("graveyard_spacing")
GRAVEYARD_COLS = _MANAGER.get("graveyard_cols")
GRAVEYARD_PLATFORM_HALF_EXTENTS = _MANAGER.get("graveyard_platform_half_extents")
GRAVEYARD_PLATFORM_HEIGHT = _MANAGER.get("graveyard_platform_height")
STOCKFISH_PATHS = _MANAGER.get("stockfish_paths")
STOCKFISH_TIME_LIMIT = _MANAGER.get("stockfish_time_limit")
LOCAL_MODEL_PATH = _MANAGER.get("local_model_path")
TILT_THRESHOLD_COS = _MANAGER.get("tilt_threshold_cos")
DISPLACEMENT_THRESHOLD = _MANAGER.get("displacement_threshold")
# Missing PIECE_HIDE_Z which is used in turn_executor.py but not in schema
PIECE_HIDE_Z = -1.0 # Default fallback if not in YAML
