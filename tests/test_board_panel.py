"""
Tests for the BoardPanel GUI component.

This module verifies that the BoardPanel initializes correctly, including its
Tkinter components and chess board representation.
"""

import tkinter as tk
from unittest.mock import MagicMock
import chess
from src.gui.board_panel import BoardPanel
from src.game_loop import GameLoop, GameState


class TestBoardPanel:
    """
    Test suite for the BoardPanel GUI component.
    """

    def test_board_panel_initialization(self):
        """
        Verify that BoardPanel initializes with correct default state and grid.
        """
        root = tk.Tk()
        loop = MagicMock(spec=GameLoop)
        board = chess.Board()
        loop.get_state.return_value = GameState(
            board, False, False, False, False, False, False, False
        )

        panel = BoardPanel(root, loop)
        assert panel.auto_play.get() is False
        assert len(panel.squares) == 64

        root.destroy()
