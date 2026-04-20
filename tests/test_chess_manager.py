"""
Tests for the ChessGameManager class.

This module verifies the chess logic managed by ChessGameManager, including
board initialization, move validation, move execution, and game state
detection (checkmate, stalemate, etc.).
"""

import chess
import pytest


class TestChessManager:
    """
    Test suite for the ChessGameManager chess logic.
    """

    def test_initial_board(self, manager):
        """
        Verify the initial board setup and move count.
        """
        assert manager.board.turn == chess.WHITE
        assert len(list(manager.board.legal_moves)) == 20

    def test_validate_move(self, manager):
        """
        Verify that various UCI moves are correctly validated.
        """
        assert manager.validate_move("e2e4") is True
        assert manager.validate_move("e2e5") is False
        assert manager.validate_move("invalid") is False
        assert manager.validate_move("e2e9") is False

    def test_push_move(self, manager):
        """
        Verify that pushing a valid move updates the board state correctly.
        """
        manager.push_move("e2e4")
        assert manager.board.turn == chess.BLACK
        assert manager.board.piece_at(chess.E4).symbol() == "P"
        assert manager.board.piece_at(chess.E2) is None

    def test_push_move_errors(self, manager):
        """
        Verify that illegal or invalid UCI moves raise ValueError.
        """
        # Illegal move
        with pytest.raises(ValueError):
            manager.push_move("e2e5")

        # Invalid UCI string
        with pytest.raises(ValueError):
            manager.push_move("invalid")

    def test_is_game_over(self, manager):
        """
        Verify game over detection via Fool's Mate.
        """
        assert manager.is_game_over() is False
        # Set board to fools mate
        manager.push_move("f2f3")
        manager.push_move("e7e5")
        manager.push_move("g2g4")
        manager.push_move("d8h4")
        assert manager.is_game_over() is True

    def test_get_ai_move(self, manager):
        """
        Verify that get_ai_move returns a valid move if Stockfish is available.
        """
        try:
            move = manager.get_ai_move()
            assert isinstance(move, chess.Move)
            assert len(move.uci()) in [4, 5]
            assert manager.validate_move(move.uci()) is True
        except RuntimeError as e:
            # Stockfish may not be installed in the CI/environment
            assert "Stockfish" in str(e)

    def test_is_game_over_stalemate(self, manager):
        """
        Verify game over detection in a stalemate scenario.
        """
        manager.board.set_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert manager.is_game_over() is True

    def test_is_game_over_insufficient_material(self, manager):
        """
        Verify game over detection with insufficient material.
        """
        manager.board.set_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
        assert manager.is_game_over() is True
