"""
Logical chess state management and Stockfish integration.

This module provides the ChessGameManager class which maintains the board state
using the python-chess library and interacts with the Stockfish chess engine
for move generation and validation. It handles the lifecycle of the engine
process and ensures proper cleanup.
"""

import chess
import chess.engine

from src.exceptions import AIEngineError
from src.config import STOCKFISH_PATHS, STOCKFISH_TIME_LIMIT


class ChessGameManager:
    """
    Manages the chess board state and the Stockfish engine.

    This class handles move validation, pushing moves to the board, and
    querying the engine for AI moves. It supports context manager usage
    for reliable engine cleanup. It acts as a bridge between the physical
    simulation and the logical chess rules.
    """

    def __init__(self):
        """
        Initialize the board and open the chess engine.

        The board is initialized to the starting chess position.
        The engine is opened using configured search paths.
        """
        self.board = chess.Board()
        self.engine = self._open_engine()

    def __enter__(self):
        """
        Context manager entry.

        Returns:
            ChessGameManager: The instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.

        Ensures the engine process is terminated when exiting the block.

        Args:
            exc_type: The type of the exception that occurred.
            exc_val: The instance of the exception that occurred.
            exc_tb: The traceback of the exception that occurred.
        """
        self.close()

    def close(self):
        """
        Close the Stockfish engine process.

        Terminates the engine subprocess and cleans up the engine attribute.
        Does nothing if the engine is already closed or not initialized.
        """
        if hasattr(self, "engine") and self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            finally:
                self.engine = None

    def validate_move(self, uci):
        """
        Check if a UCI move is syntactically valid and legal.

        This method validates the move string format and checks if the move
        is allowed in the current board state according to chess rules.

        Args:
            uci: String representing the move in Universal Chess Interface format.

        Returns:
            bool: True if the move is legal, False otherwise.
        """
        try:
            move = chess.Move.from_uci(uci)
            return move in self.board.legal_moves
        except ValueError:
            return False

    def is_game_over(self):
        """
        Check if the game has ended.

        Determines if the current board state is a terminal state (checkmate,
        stalemate, draw, etc.).

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def push_move(self, uci):
        """
        Apply a legal move to the board.

        Updates the internal board state by applying the specified UCI move.

        Args:
            uci: String representing the move in UCI format.
        """
        self.board.push_uci(uci)

    def _open_engine(self):
        """
        Open a Stockfish process from configured paths.

        Attempts to locate and start the Stockfish engine using a list of
        predefined paths.

        Returns:
            chess.engine.SimpleEngine: A connected chess engine instance.

        Raises:
            AIEngineError: If Stockfish cannot be found or started in any path.
        """
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

    def get_ai_move(self, limit_time=STOCKFISH_TIME_LIMIT):
        """
        Get the best move from the AI engine.

        Queries the Stockfish engine to determine the best move for the
        current board state within the specified time limit.

        Args:
            limit_time: Time limit for the engine's search in seconds.

        Returns:
            chess.Move: The best move suggested by the engine.

        Raises:
            AIEngineError: If the engine fails to provide a move or errors.
        """
        try:
            result = self.engine.play(self.board, chess.engine.Limit(time=limit_time))
            if not result.move:
                raise AIEngineError("Stockfish failed to suggest a move.")
            return result.move
        except chess.engine.EngineError as e:
            raise AIEngineError(f"Stockfish engine error during play: {e}")
