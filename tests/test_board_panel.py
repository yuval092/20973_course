import pytest
import tkinter as tk
from unittest.mock import MagicMock
from src.gui.board_panel import BoardPanel
from src.game_loop import GameLoop, GameState
import chess

def test_board_panel_initialization():
    root = tk.Tk()
    loop = MagicMock(spec=GameLoop)
    board = chess.Board()
    loop.get_state.return_value = GameState(board, False, False, False, False, False, False, False)
    
    panel = BoardPanel(root, loop)
    assert panel.auto_play.get() is False
    assert len(panel.squares) == 64
    
    # Explicit cleanup to prevent Variable.__del__ RuntimeError in pytest
    panel.destroy()
    root.update()
    root.destroy()
