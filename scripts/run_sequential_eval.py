"""
Sequential game suite (Section 11.1).
Runs complete game traces using Stockfish and the learned policy.
"""

import argparse
import logging
import chess
import numpy as np
from sb3_contrib import TQC
from src.game_runtime import bootstrap_game_systems
from src.logic.operation_planner import OperationPlanner

logger = logging.getLogger(__name__)

def run_sequential_eval(model_path, n_games=5):
    # This requires more complex setup to disable teleport fallback
    # For now, this is a skeleton.
    print(f"Skeleton for sequential eval using model {model_path}")
    print(f"Will run {n_games} games.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-games", type=int, default=5)
    args = parser.parse_args()
    run_sequential_eval(args.checkpoint, args.n_games)
