"""Translate validated chess moves into physical pick-and-place operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess


@dataclass
class PickPlaceOp:
    """One physical move of one MuJoCo piece body."""

    target_square: str
    dest_square: str
    piece_name: str
    is_capture: bool = False
    promotion: Optional[int] = None


class OperationPlanner:
    """Plan the ordered manipulation ops for a single validated chess move."""

    @staticmethod
    def _require_piece(square_to_piece: dict[str, str], square_name: str) -> str:
        """Return the mapped MuJoCo piece for a square or raise clearly."""
        try:
            return square_to_piece[square_name]
        except KeyError as exc:
            raise KeyError(f"Missing physical piece mapping for square '{square_name}'.") from exc

    def generate_operations(
        self,
        move: chess.Move,
        board: chess.Board,
        square_to_piece: dict[str, str],
    ) -> list[PickPlaceOp]:
        """Generate the controller-visible ops required to execute ``move``.

        The planner inspects the pre-move board state. Special moves expand
        into multiple physical operations so the runtime can evacuate captured
        pieces first, then move the active piece into the final destination.
        """
        ops: list[PickPlaceOp] = []

        if board.is_castling(move):
            src_sq = chess.square_name(move.from_square)
            dest_sq = chess.square_name(move.to_square)
            ops.append(PickPlaceOp(target_square=src_sq, dest_square=dest_sq, piece_name=self._require_piece(square_to_piece, src_sq)))

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
                captured_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                target_sq_name = chess.square_name(captured_square)
            else:
                captured_square = move.to_square
                target_sq_name = chess.square_name(captured_square)

            captured_piece = board.piece_at(captured_square)
            if captured_piece is None:
                raise ValueError(
                    f"Capture move {move.uci()} expected a piece on {target_sq_name}, but the logical board had none."
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
