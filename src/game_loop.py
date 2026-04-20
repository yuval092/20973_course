"""
Decoupled orchestrator for a RoboChess game.

This module provides the GameLoop class, which coordinates between the 
chess logic, operation planning, and physical execution.
"""

import chess
from src.exceptions import AIEngineError
from src.models import GameState, MoveResult
from src.game_runtime import GameOrchestrator


class GameLoop:
    """Orchestrates the flow of a RoboChess game."""
    
    def __init__(self, systems):
        """
        Initialize the game loop with system components.
        
        Args:
            systems: The GameSystems instance.
        """
        self.systems = systems
        
    def get_state(self):
        """
        Get the current logical state of the game.
        
        Returns:
            A GameState instance.
        """
        board = self.systems.manager.board
        return GameState(
            board=board,
            is_game_over=board.is_game_over(claim_draw=True),
            is_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            is_insufficient_material=board.is_insufficient_material(),
            is_fifty_moves=board.can_claim_fifty_moves(),
            is_threefold=board.can_claim_threefold_repetition(),
        )

    def request_hint(self):
        """
        Query Stockfish for the best move in the current position.
        
        Returns:
            The best move in UCI format.
        """
        best_move = self.systems.manager.get_ai_move()
        return best_move.uci()
        
    def submit_move(self, uci, viewer=None):
        """
        Execute a human move.
        
        Args:
            uci: The move in UCI format.
            viewer: Optional MuJoCo viewer instance.
            
        Returns:
            A MoveResult instance.
        """
        if not self.systems.manager.validate_move(uci):
            return MoveResult(False, f"Illegal move: {uci}")
            
        move = chess.Move.from_uci(uci)
        ops = self.systems.planner.generate_operations(
            move, self.systems.manager.board, self.systems.square_to_piece)
        try:
            GameOrchestrator.execute_ops(
                self.systems.mj_model,
                self.systems.mj_data,
                ops,
                self.systems.controller,
                self.systems.square_to_piece,
                systems=self.systems,
                viewer=viewer,
                physical_arm=True
            )
            self.systems.manager.push_move(uci)
            return MoveResult(True, f"Moved {uci}")
        except Exception as e:
            return MoveResult(False, f"Execution failed: {e}", error=e)

    def execute_ai_turn(self, viewer=None):
        """
        Execute an AI turn.
        
        Args:
            viewer: Optional MuJoCo viewer instance.
            
        Returns:
            A MoveResult instance.
        """
        try:
            move = self.systems.manager.get_ai_move()
            if not move:
                raise AIEngineError("AI returned no move.")
            
            ops = self.systems.planner.generate_operations(
                move, self.systems.manager.board, self.systems.square_to_piece)
            
            GameOrchestrator.execute_ops(
                self.systems.mj_model,
                self.systems.mj_data,
                ops,
                self.systems.controller,
                self.systems.square_to_piece,
                systems=self.systems,
                viewer=viewer,
                physical_arm=True
            )
            self.systems.manager.push_move(move.uci())
            return MoveResult(True, f"AI played: {move.uci()}")
        except Exception as e:
            return MoveResult(False, f"AI Execution failed: {e}", error=e)
