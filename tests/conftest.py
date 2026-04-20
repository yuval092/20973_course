"""
Shared pytest fixtures and configuration for the test suite.

This file provides common fixtures used across multiple test modules to ensure
consistency and reduce code duplication.
"""

import sys
from pathlib import Path
import pytest
from src.logic.chess_manager import ChessGameManager

# Add the project root to the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def manager():
    """
    Fixture to provide a ChessGameManager that is closed after the test.

    Returns:
        ChessGameManager: An initialized chess game manager.
    """
    m = ChessGameManager()
    yield m
    m.close()
