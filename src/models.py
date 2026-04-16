from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import mujoco
import chess

# Type alias for square name (e.g., "e2") to MuJoCo body name (e.g., "w_pawn_5")
SquareMap = Dict[str, str]

@dataclass
class GameSystems:
    """Collection of initialized core system components."""
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    env: Any
    rl_model: Any
    manager: Any
    planner: Any
    controller: Any
    square_to_piece: SquareMap
    arm_home_grip: Any
    check_registry: Any
    # Counter for captured pieces to arrange them in a grid in the graveyard
    captured_count: Dict[str, int] = None

    def __post_init__(self):
        if self.captured_count is None:
            self.captured_count = {"white": 0, "black": 0}

@dataclass
class MoveResult:
    """Outcome of a turn execution."""
    success: bool
    message: str
    error: Optional[Exception] = None

@dataclass
class GameState:
    """Snapshot of the logical game state for the GUI."""
    board: chess.Board
    is_game_over: bool
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    is_insufficient_material: bool
    is_fifty_moves: bool
    is_threefold: bool
