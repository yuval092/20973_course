"""
Waypoint and scenario transition logic for RoboChess.

Defines the vertical targets (Z-levels) for different movement scenarios
and validates the transitions between them in multi-step task chains.
"""
import numpy as np
from src.utils.config import load_config

_cfg = load_config("env")
SAFE_Z  = _cfg["safe_z"]    # 0.550m
GRASP_Z = _cfg["grasp_z"]   # 0.430m
HOVER_Z = _cfg.get("hover_z", 0.460)

SCENARIO_EXIT_Z = {
    "transit": SAFE_Z,
    "descend": HOVER_Z,
    "ascend":  SAFE_Z,
}
SCENARIO_ENTRY_Z = {
    "transit": SAFE_Z,
    "descend": SAFE_Z,
    "ascend":  HOVER_Z,
}

VALID_TRANSITIONS = {
    ("transit", "transit"): True,
    ("transit", "descend"): True,
    ("descend", "ascend"):  True,
    ("ascend",  "transit"): True,
    ("ascend",  "descend"): True,
}

CHAIN_SHORTCUTS = {
    "full_move": ["transit", "descend", "ascend", "transit", "descend", "ascend"],
    "pick":      ["transit", "descend", "ascend"],
    "place":     ["transit", "descend", "ascend"],
    "vertical":  ["descend", "ascend"],
}

def validate_chain(chain: list[str]) -> None:
    """
    Validates that a sequence of scenarios forms a valid physical transition chain.
    
    Args:
        chain: List of scenario names (e.g., ['transit', 'descend']).
        
    Raises:
        ValueError: If an invalid transition is detected.
    """
    for i in range(len(chain) - 1):
        key = (chain[i], chain[i + 1])
        if not VALID_TRANSITIONS.get(key, False):
            raise ValueError(f"Invalid chain transition at pos {i}: {chain[i]} -> {chain[i+1]}")

def derive_goal_pos(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """
    Returns the 3D target position for a given scenario and 2D cell coordinate.
    
    Args:
        scenario: The movement phase ('transit', 'descend', 'ascend').
        cell_xy: The 2D (X, Y) board coordinate.
        
    Returns:
        A 3D NumPy array [X, Y, Z].
        
    Raises:
        ValueError: If the scenario is unknown.
    """
    z_map = {"transit": SAFE_Z, "descend": HOVER_Z, "ascend": SAFE_Z}
    if scenario not in z_map:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.array([cell_xy[0], cell_xy[1], z_map[scenario]])

def exit_waypoint(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """
    Returns the 3D exit waypoint for a scenario, used as the entry point for the next.
    
    Args:
        scenario: The completed scenario.
        cell_xy: The 2D (X, Y) coordinate at the end of the scenario.
        
    Returns:
        A 3D NumPy array [X, Y, Z].
        
    Raises:
        ValueError: If the scenario is unknown.
    """
    if scenario not in SCENARIO_EXIT_Z:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.array([cell_xy[0], cell_xy[1], SCENARIO_EXIT_Z[scenario]])
