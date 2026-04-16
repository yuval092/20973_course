"""
EpisodeSampler (Section 9)
──────────────────────────
Generates manipulation episodes for the training environment.
"""

from dataclasses import dataclass
import numpy as np
import chess
from src.config import TABLE_HEIGHT, PIECE_NAMES
from src.control.execution_controller import ExecutionController

@dataclass
class Episode:
    piece_name:  str          # MuJoCo body name, e.g. "w_pawn_3"
    src_square:  str          # UCI square, e.g. "e2"
    dst_square:  str          # UCI square, e.g. "e4"
    src_pos:     np.ndarray   # MuJoCo XYZ of source square
    dest_pos:    np.ndarray   # MuJoCo XYZ of destination square
    is_capture:  bool
    piece_type:  str          # "pawn", "rook", etc.


REGIONS = {
    1: ["d4", "e4", "d5", "e5"],
    2: [f"{f}{r}" for f in "cdef" for r in "3456"],   # 4×4 = 16 squares
    3: [f"{f}{r}" for f in "abcdefgh" for r in "12345678"],  # all 64
}

class EpisodeSampler:
    def __init__(self, rng: np.random.Generator, curriculum_stage: int):
        self._rng   = rng
        self._stage = curriculum_stage

    def sample(self) -> Episode:
        # For stage 1-2, restrict to regions and specific piece types
        if self._stage == 1:
            return self._sample_stage_1()
        elif self._stage == 2:
            return self._sample_stage_2()
        
        bucket = self._rng.choice(["legal", "adversarial"], p=[0.70, 0.30])
        if bucket == "legal":
            return self._sample_legal()
        else:
            return self._sample_adversarial()

    def _sample_stage_1(self) -> Episode:
        """Stage 1: Central single-piece, pawns only."""
        src = self._rng.choice(REGIONS[1])
        dst = self._rng.choice([s for s in REGIONS[1] if s != src])
        # Force a white pawn for stage 1 simplicity
        piece_name = f"w_pawn_{self._rng.integers(1, 9)}"
        return Episode(
            piece_name = piece_name,
            src_square = src,
            dst_square = dst,
            src_pos    = ExecutionController.get_square_pos(src),
            dest_pos   = ExecutionController.get_square_pos(dst),
            is_capture = False,
            piece_type = "pawn",
        )

    def _sample_stage_2(self) -> Episode:
        """Stage 2: Central region, Pawn, Knight."""
        src = self._rng.choice(REGIONS[2])
        dst = self._rng.choice([s for s in REGIONS[2] if s != src])
        p_type = self._rng.choice(["pawn", "knight"])
        color = self._rng.choice(["w", "b"])
        if p_type == "pawn":
            piece_name = f"{color}_pawn_{self._rng.integers(1, 9)}"
        else:
            piece_name = f"{color}_knight_{self._rng.integers(1, 3)}"
            
        return Episode(
            piece_name = piece_name,
            src_square = src,
            dst_square = dst,
            src_pos    = ExecutionController.get_square_pos(src),
            dest_pos   = ExecutionController.get_square_pos(dst),
            is_capture = False,
            piece_type = p_type,
        )

    def _sample_legal(self) -> Episode:
        """Sample from a random legal board position."""
        from src.env.board_state_gen import sample_board_state
        board, piece_map = sample_board_state(self._rng, stage=self._stage)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # If no legal moves, try again with a fresh board
            return self._sample_legal()

        move = self._rng.choice(legal_moves)
        src  = chess.square_name(move.from_square)
        dst  = chess.square_name(move.to_square)

        piece = board.piece_at(move.from_square)
        piece_type = chess.piece_name(piece.piece_type)
        color_char = "w" if piece.color == chess.WHITE else "b"
        
        piece_name = piece_map.get((color_char, piece_type, src))
        if piece_name is None:
            return self._sample_legal()

        return Episode(
            piece_name = piece_name,
            src_square = src,
            dst_square = dst,
            src_pos    = ExecutionController.get_square_pos(src),
            dest_pos   = ExecutionController.get_square_pos(dst),
            is_capture = board.is_capture(move),
            piece_type = piece_type,
        )

    def _sample_adversarial(self) -> Episode:
        """Sample adversarial patterns (edge files, etc)."""
        # For now, just fallback to legal.
        return self._sample_legal()
