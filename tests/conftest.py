import sys
from pathlib import Path
import pytest
from src.logic.chess_manager import ChessGameManager


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def manager():
    """Fixture to provide a ChessGameManager that is closed after the test."""
    m = ChessGameManager()
    yield m
    m.close()
