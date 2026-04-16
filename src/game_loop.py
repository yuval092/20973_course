from typing import Optional
import chess

from src.exceptions import AIEngineError, ExecutionError
from src.models import GameSystems, GameState, MoveResult

class GameLoop:
    """Decoupled orchestrator for a RoboChess game."""
    
    def __init__(self, systems: GameSystems):
        self.systems = systems
        
    def get_state(self) -> GameState:
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

    def request_hint(self) -> str:
        """Query Stockfish for the best move in the current position."""
        best_move = self.systems.manager.get_ai_move()
        return best_move.uci()
        
    def submit_move(self, uci: str, viewer=None) -> MoveResult:
        """Execute a human move."""
        if not self.systems.manager.validate_move(uci):
            return MoveResult(False, f"Illegal move: {uci}")
            
        move = chess.Move.from_uci(uci)
        ops = self.systems.planner.generate_operations(move, self.systems.manager.board, self.systems.square_to_piece)
        try:
            from src.game_runtime import execute_human_ops
            execute_human_ops(
                self.systems.mj_model,
                self.systems.mj_data,
                ops,
                self.systems.controller,
                self.systems.square_to_piece,
                systems=self.systems,
                viewer=viewer,
            )
            self.systems.manager.push_move(uci)
            return MoveResult(True, f"Moved {uci}")
        except Exception as e:
            return MoveResult(False, f"Execution failed: {e}", error=e)

    def execute_ai_turn(self, viewer=None) -> MoveResult:
        """Execute an AI turn."""
        try:
            move = self.systems.manager.get_ai_move()
            if not move:
                raise AIEngineError("AI returned no move.")
            
            ops = self.systems.planner.generate_operations(move, self.systems.manager.board, self.systems.square_to_piece)
            from src.game_runtime import execute_ai_ops
            execute_ai_ops(
                ops,
                self.systems.controller,
                self.systems.square_to_piece,
                self.systems.mj_model,
                self.systems.mj_data,
                viewer,
                systems=self.systems,
            )
            self.systems.manager.push_move(move.uci())
            return MoveResult(True, f"AI played: {move.uci()}")
        except Exception as e:
            return MoveResult(False, f"AI Execution failed: {e}", error=e)
