"""
Tests for the GameLoop class.

This module verifies the main game loop's ability to provide current game state
and handle move submissions, including validation of illegal moves.
"""

from unittest.mock import MagicMock
import chess
from src.game_loop import GameLoop


class TestGameLoop:
    """
    Test suite for the GameLoop orchestration logic.
    """

    def test_game_loop_get_state(self):
        """
        Verify that GameLoop correctly reports the current board state.
        """
        systems = MagicMock()
        systems.manager = MagicMock()
        board = chess.Board()
        systems.manager.board = board

        loop = GameLoop(systems)
        state = loop.get_state()
        assert state.board == board
        assert state.is_game_over is False
        assert state.is_checkmate is False

    def test_submit_move_invalid(self):
        """
        Verify that submitting an invalid move returns a failure result.
        """
        systems = MagicMock()
        systems.manager.validate_move.return_value = False
        loop = GameLoop(systems)

        res = loop.submit_move("e2e5")
        assert res.success is False
        assert "Illegal" in res.message
