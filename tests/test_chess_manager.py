import chess
import pytest
from src.logic.chess_manager import ChessGameManager

def test_initial_board():
    manager = ChessGameManager()
    assert manager.board.turn == chess.WHITE
    assert len(list(manager.board.legal_moves)) == 20

def test_validate_move():
    manager = ChessGameManager()
    assert manager.validate_move("e2e4") is True
    assert manager.validate_move("e2e5") is False
    assert manager.validate_move("invalid") is False
    assert manager.validate_move("e2e9") is False

def test_push_move():
    manager = ChessGameManager()
    manager.push_move("e2e4")
    assert manager.board.turn == chess.BLACK
    assert manager.board.piece_at(chess.E4).symbol() == "P"
    assert manager.board.piece_at(chess.E2) is None

def test_push_move_errors():
    manager = ChessGameManager()
    # Illegal move
    with pytest.raises(ValueError):
        manager.push_move("e2e5")
    
    # Invalid UCI string
    with pytest.raises(ValueError):
        manager.push_move("invalid")

def test_is_game_over():
    manager = ChessGameManager()
    assert manager.is_game_over() is False
    # Set board to fools mate
    manager.push_move("f2f3")
    manager.push_move("e7e5")
    manager.push_move("g2g4")
    manager.push_move("d8h4")
    assert manager.is_game_over() is True

def test_get_ai_move():
    manager = ChessGameManager()
    # Since stockfish is an external dependency, we just make sure
    # get_ai_move returns a valid string (like "e7e5") or raises if Stockfish isn't installed.
    # The actual behavior depends on the system, so we catch potential errors or assert it returns a move.
    try:
        move = manager.get_ai_move()
        assert isinstance(move, chess.Move)
        assert len(move.uci()) in [4, 5]
        assert manager.validate_move(move.uci()) is True
    except RuntimeError as e:
        # Stockfish may not be installed in the CI/environment
        assert "Stockfish" in str(e)

