import random
import chess
import numpy as np
from src.config import PIECE_NAMES

def sample_board_state(rng: np.random.Generator, stage: int = 1):
    """
    Generate a random legal board state and a mapping from 
    (color, type, square) to MuJoCo piece body names.
    
    In earlier curriculum stages, we might want simpler boards.
    """
    board = chess.Board()
    
    if stage <= 3:
        # Simple board: just pieces in their starting positions
        # but we might want to sample a few random moves
        # For stage 1-3, often we just need the piece to be on its nominal square.
        # But let's make it a bit more interesting.
        pass
    else:
        # Sample a random number of legal moves to get a diverse state
        n_moves = rng.integers(0, 40)
        for _ in range(n_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))

    # Create a mapping from (color_char, piece_type_str, square_name) to body_name
    # color_char: 'w' or 'b'
    # piece_type_str: 'pawn', 'rook', etc.
    piece_map = {}
    
    # We need to track which body names we've assigned to which piece on the board.
    # MuJoCo bodies are named like "w_pawn_1", "w_rook_1", "w_king", etc.
    available_bodies = list(PIECE_NAMES)
    random.shuffle(available_bodies)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_char = 'w' if piece.color == chess.WHITE else 'b'
            p_type_str = chess.piece_name(piece.piece_type)
            sq_name = chess.square_name(square)
            
            # Find a matching body name
            body_name = None
            prefix = f"{color_char}_{p_type_str}"
            for b in available_bodies:
                if b.startswith(prefix):
                    body_name = b
                    available_bodies.remove(b)
                    break
            
            if body_name:
                piece_map[(color_char, p_type_str, sq_name)] = body_name
                
    return board, piece_map
