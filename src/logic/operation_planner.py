"""
Translate validated chess moves into physical pick-and-place operations.

This module provides the OperationPlanner class, which breaks down high-level
chess moves (including special moves like castling and en passant) into a
sequence of physical robot operations. Each operation is represented by a
PickPlaceOp instance.
"""

import chess


class PickPlaceOp:
    """
    One physical move of one MuJoCo piece body.

    This class represents a single pick-and-place operation, describing
    where a piece starts, where it should go, and whether it's a capture.
    """

    def __init__(self, target_square, dest_square, piece_name, is_capture=False, promotion=None):
        """
        Initialize a PickPlaceOp instance.

        Args:
            target_square: The square the piece is currently on (e.g., 'e2').
            dest_square: The square or graveyard the piece is moving to.
            piece_name: The MuJoCo body name of the piece to be moved.
            is_capture: Whether this operation involves capturing a piece.
            promotion: The piece type to promote to if applicable.
        """
        self.target_square = target_square
        self.dest_square = dest_square
        self.piece_name = piece_name
        self.is_capture = is_capture
        self.promotion = promotion


class OperationPlanner:
    """
    Plan the ordered manipulation ops for a single validated chess move.

    This class converts a chess.Move into one or more PickPlaceOp objects
    that the execution controller can process.
    """

    def _require_piece(self, square_to_piece, square_name):
        """
        Return the mapped MuJoCo piece for a square or raise clearly.

        Args:
            square_to_piece: Dictionary mapping square names to piece names.
            square_name: The name of the square to look up.

        Returns:
            str: The MuJoCo piece name.

        Raises:
            KeyError: If no piece is mapped to the given square.
        """
        try:
            return square_to_piece[square_name]
        except KeyError as exc:
            raise KeyError(f"Missing physical piece mapping for square '{square_name}'.") from exc

    def generate_operations(self, move, board, square_to_piece):
        """
        Generate the controller-visible ops required to execute a move.

        The planner inspects the pre-move board state. Special moves expand
        into multiple physical operations so the runtime can evacuate captured
        pieces first, then move the active piece into the final destination.

        Args:
            move: The chess.Move to execute.
            board: The current chess.Board state.
            square_to_piece: Dictionary mapping squares to physical pieces.

        Returns:
            list: A list of PickPlaceOp instances.
        """
        ops = []

        if board.is_castling(move):
            src_sq = chess.square_name(move.from_square)
            dest_sq = chess.square_name(move.to_square)
            ops.append(
                PickPlaceOp(
                    target_square=src_sq,
                    dest_square=dest_sq,
                    piece_name=self._require_piece(square_to_piece, src_sq)
                )
            )

            if board.is_kingside_castling(move):
                rook_src = chess.square(7, chess.square_rank(move.from_square))
                rook_dest = chess.square(5, chess.square_rank(move.from_square))
            else:
                rook_src = chess.square(0, chess.square_rank(move.from_square))
                rook_dest = chess.square(3, chess.square_rank(move.from_square))

            rook_src_name = chess.square_name(rook_src)
            ops.append(
                PickPlaceOp(
                    target_square=rook_src_name,
                    dest_square=chess.square_name(rook_dest),
                    piece_name=self._require_piece(square_to_piece, rook_src_name),
                )
            )
            return ops

        if board.is_capture(move):
            # En passant captures the pawn beside the destination square,
            # not the piece physically located on the destination square.
            if board.is_en_passant(move):
                captured_square = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square)
                )
                target_sq_name = chess.square_name(captured_square)
            else:
                captured_square = move.to_square
                target_sq_name = chess.square_name(captured_square)

            captured_piece = board.piece_at(captured_square)
            if captured_piece is None:
                raise ValueError(
                    f"Capture move {move.uci()} expected a piece on {target_sq_name}, "
                    "but the logical board had none."
                )
            color_name = "white" if captured_piece.color == chess.WHITE else "black"
            graveyard = f"{color_name}_graveyard"

            ops.append(
                PickPlaceOp(
                    target_square=target_sq_name,
                    dest_square=graveyard,
                    piece_name=self._require_piece(square_to_piece, target_sq_name),
                    is_capture=True,
                )
            )

        src_sq = chess.square_name(move.from_square)
        dest_sq = chess.square_name(move.to_square)
        ops.append(
            PickPlaceOp(
                target_square=src_sq,
                dest_square=dest_sq,
                piece_name=self._require_piece(square_to_piece, src_sq),
                promotion=move.promotion,
            )
        )
        return ops
