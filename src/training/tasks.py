"""Task-generation helpers for RoboChess manipulation fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import chess


@dataclass(frozen=True)
class ManipulationTask:
    """A single chess-manipulation task derived from a legal move."""

    fen: str
    move_uci: str
    piece_name: str
    source_square: str
    dest_square: str
    is_capture: bool
    promotion: int | None = None


def task_from_move(board: chess.Board, move: chess.Move, square_to_piece: dict[str, str]) -> ManipulationTask:
    """Build a fine-tuning task description from a legal move and mapping."""
    source_square = chess.square_name(move.from_square)
    dest_square = chess.square_name(move.to_square)
    return ManipulationTask(
        fen=board.fen(),
        move_uci=move.uci(),
        piece_name=square_to_piece[source_square],
        source_square=source_square,
        dest_square=dest_square,
        is_capture=board.is_capture(move),
        promotion=move.promotion,
    )


class TaskSampler:
    """Generate deterministic or sampled manipulation tasks from legal chess states."""

    def __init__(self, opening_sequences: Sequence[Sequence[str]] | None = None):
        self.opening_sequences = tuple(tuple(seq) for seq in (opening_sequences or ()))

    def board_from_sequence(self, moves: Iterable[str]) -> chess.Board:
        """Construct a board by replaying UCI moves from the initial position."""
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        return board

    def sample_first_legal_task(self, board: chess.Board, square_to_piece: dict[str, str]) -> ManipulationTask:
        """Return the first legal move as a deterministic task."""
        move = next(iter(board.legal_moves))
        return task_from_move(board, move, square_to_piece)

    def tasks_from_openings(self, square_to_piece: dict[str, str]) -> list[ManipulationTask]:
        """Generate one task per configured opening continuation."""
        tasks: list[ManipulationTask] = []
        for sequence in self.opening_sequences:
            board = self.board_from_sequence(sequence)
            tasks.append(self.sample_first_legal_task(board, square_to_piece))
        return tasks
