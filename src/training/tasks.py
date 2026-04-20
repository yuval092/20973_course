"""
Task-generation helpers for RoboChess manipulation fine-tuning.

This module provides tools for generating specific chess manipulation tasks
from legal moves. It includes the ManipulationTask data structure and the
TaskSampler class for creating tasks from board states or opening sequences.
"""

import chess


class ManipulationTask:
    """
    A single chess-manipulation task derived from a legal move.

    This class encapsulates all the information needed by the environment
    to set up and execute a specific piece movement task.
    """

    def __init__(self, fen, move_uci, piece_name, source_square, dest_square,
                 is_capture, promotion=None):
        """
        Initialize a ManipulationTask instance.

        Args:
            fen: The FEN string representing the board state.
            move_uci: The move to be executed in UCI format.
            piece_name: The MuJoCo name of the piece to move.
            source_square: The square the piece starts on.
            dest_square: The square the piece moves to.
            is_capture: Whether the move captures another piece.
            promotion: The piece type to promote to, if any.
        """
        self.fen = fen
        self.move_uci = move_uci
        self.piece_name = piece_name
        self.source_square = source_square
        self.dest_square = dest_square
        self.is_capture = is_capture
        self.promotion = promotion


def task_from_move(board, move, square_to_piece):
    """
    Build a fine-tuning task description from a legal move and mapping.

    Args:
        board: The current chess.Board.
        move: The chess.Move to convert.
        square_to_piece: Dictionary mapping square names to piece names.

    Returns:
        ManipulationTask: The generated task object.
    """
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
    """
    Generate deterministic or sampled manipulation tasks from chess states.

    This class handles the creation of tasks from predefined opening sequences
    or by sampling legal moves from a given board position.
    """

    def __init__(self, opening_sequences=None):
        """
        Initialize the TaskSampler.

        Args:
            opening_sequences: Optional list of move sequences (lists of UCI strings).
        """
        if opening_sequences is None:
            opening_sequences = ()
        self.opening_sequences = tuple(tuple(seq) for seq in opening_sequences)

    def board_from_sequence(self, moves):
        """
        Construct a board by replaying UCI moves from the initial position.

        Args:
            moves: An iterable of UCI move strings.

        Returns:
            chess.Board: The resulting board state.
        """
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        return board

    def sample_first_legal_task(self, board, square_to_piece):
        """
        Return the first legal move as a deterministic task.

        Args:
            board: The chess.Board to sample from.
            square_to_piece: Mapping of squares to physical piece names.

        Returns:
            ManipulationTask: The first legal move as a task.
        """
        move = next(iter(board.legal_moves))
        return task_from_move(board, move, square_to_piece)

    def tasks_from_openings(self, square_to_piece):
        """
        Generate one task per configured opening continuation.

        Args:
            square_to_piece: Mapping of squares to physical piece names.

        Returns:
            list: A list of ManipulationTask instances.
        """
        tasks = []
        for sequence in self.opening_sequences:
            board = self.board_from_sequence(sequence)
            tasks.append(self.sample_first_legal_task(board, square_to_piece))
        return tasks
