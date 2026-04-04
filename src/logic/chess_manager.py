"""Logical chess state management and Stockfish integration."""

from __future__ import annotations

import chess
import chess.engine

from src.exceptions import AIEngineError
from src.config import STOCKFISH_PATHS, STOCKFISH_TIME_LIMIT


class ChessGameManager:
    """Own the python-chess board and AI move generation."""

    def __init__(self):
        self.board = chess.Board()

    def validate_move(self, uci: str) -> bool:
        """Return True when a UCI move is syntactically valid and legal."""
        try:
            move = chess.Move.from_uci(uci)
            return move in self.board.legal_moves
        except ValueError:
            return False

    def is_game_over(self) -> bool:
        """Return True if the game has ended."""
        return self.board.is_game_over()

    def push_move(self, uci: str) -> None:
        """Push a legal move in UCI format onto the board."""
        self.board.push_uci(uci)

    def _open_engine(self) -> chess.engine.SimpleEngine:
        """Open a Stockfish process from the configured search paths."""
        engine = None
        for path in STOCKFISH_PATHS:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(path)
                break
            except (FileNotFoundError, OSError, chess.engine.EngineError):
                continue

        if not engine:
            raise AIEngineError("Stockfish engine not found in search paths.")
        return engine

    def get_ai_move(self, limit_time: float = STOCKFISH_TIME_LIMIT) -> chess.Move:
        """Get the best move from Stockfish for the current position."""
        engine = self._open_engine()
        try:
            result = engine.play(self.board, chess.engine.Limit(time=limit_time))
            if not result.move:
                raise AIEngineError("Stockfish failed to suggest a move.")
            return result.move
        except chess.engine.EngineError as e:
            raise AIEngineError(f"Stockfish engine error during play: {e}")
        finally:
            try:
                engine.quit()
            except Exception:
                pass
