"""
Core data models and state containers for RoboChess.

This module defines classes for managing system components, game state,
and turn execution results.
"""


class GameSystems:
    """Collection of initialized core system components."""

    def __init__(self, mj_model, mj_data, env, rl_model, manager, planner,
                 controller, square_to_piece, arm_home_grip, check_registry,
                 captured_count=None):
        """
        Initialize the system components container.
        
        Args:
            mj_model: The MuJoCo model object.
            mj_data: The MuJoCo data object.
            env: The chess pick and place environment.
            rl_model: The trained reinforcement learning policy.
            manager: The high-level chess game manager.
            planner: The move operation planner.
            controller: The robotic execution controller.
            square_to_piece: Mapping of board squares to piece body names.
            arm_home_grip: The home position for the robot arm.
            check_registry: Registry of runtime validation checks.
            captured_count: Dictionary tracking captured pieces for each color.
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.env = env
        self.rl_model = rl_model
        self.manager = manager
        self.planner = planner
        self.controller = controller
        self.square_to_piece = square_to_piece
        self.arm_home_grip = arm_home_grip
        self.check_registry = check_registry
        
        if captured_count is None:
            self.captured_count = {"white": 0, "black": 0}
        else:
            self.captured_count = captured_count


class MoveResult:
    """Outcome of a turn execution."""

    def __init__(self, success, message, error=None):
        """
        Initialize a move execution result.
        
        Args:
            success: Boolean indicating if the move was successful.
            message: Descriptive message about the outcome.
            error: Optional exception if an error occurred.
        """
        self.success = success
        self.message = message
        self.error = error


class GameState:
    """Snapshot of the logical game state for the GUI."""

    def __init__(self, board, is_game_over, is_check, is_checkmate,
                 is_stalemate, is_insufficient_material, is_fifty_moves,
                 is_threefold):
        """
        Initialize a game state snapshot.
        
        Args:
            board: The current chess board object.
            is_game_over: True if the game has ended.
            is_check: True if the current king is in check.
            is_checkmate: True if the game ended in checkmate.
            is_stalemate: True if the game ended in stalemate.
            is_insufficient_material: True if the game ended due to lack of material.
            is_fifty_moves: True if the 50-move rule applies.
            is_threefold: True if threefold repetition applies.
        """
        self.board = board
        self.is_game_over = is_game_over
        self.is_check = is_check
        self.is_checkmate = is_checkmate
        self.is_stalemate = is_stalemate
        self.is_insufficient_material = is_insufficient_material
        self.is_fifty_moves = is_fifty_moves
        self.is_threefold = is_threefold
