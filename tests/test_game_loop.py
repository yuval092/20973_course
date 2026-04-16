import pytest
import chess
from unittest.mock import MagicMock
from src.game_loop import GameLoop, MoveResult
from src.game_runtime import GameSystems

def test_game_loop_get_state():
    systems = MagicMock(spec=GameSystems)
    systems.manager = MagicMock()
    board = chess.Board()
    systems.manager.board = board
    
    loop = GameLoop(systems)
    state = loop.get_state()
    assert state.board == board
    assert state.is_game_over is False
    assert state.is_checkmate is False
    
def test_submit_move_invalid():
    systems = MagicMock()
    systems.manager.validate_move.return_value = False
    loop = GameLoop(systems)
    
    res = loop.submit_move("e2e5")
    assert res.success is False
    assert "Illegal" in res.message
