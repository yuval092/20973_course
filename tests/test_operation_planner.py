"""
Unit tests for the operation planner.

This module verifies that the OperationPlanner correctly translates chess moves
into a sequence of robot arm operations (PickPlaceOp), including captures, 
castling, and promotions.
"""

import chess
import pytest
from src.logic.operation_planner import OperationPlanner


class TestOperationPlanner:
    """
    Test suite for translating chess moves into robot arm operations.
    """

    def _get_dummy_mapping(self):
        """
        Create a full board mapping of squares to dummy piece names.
        """
        mapping = {}
        for file_idx in range(8):
            for rank_idx in range(8):
                sq = chess.square_name(chess.square(file_idx, rank_idx))
                mapping[sq] = f"piece_{sq}"
        return mapping

    def test_standard_move_ops(self):
        """
        Verify that a standard non-capture move generates a single operation.
        """
        planner = OperationPlanner()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        mapping = self._get_dummy_mapping()
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 1
        assert ops[0].target_square == "e2"
        assert ops[0].dest_square == "e4"
        assert ops[0].piece_name == "piece_e2"
        assert ops[0].is_capture is False

    def test_capture_move_ops(self):
        """
        Verify that a capture move generates two operations: graveyard move then piece move.
        """
        planner = OperationPlanner()
        # Set up a board with a possible capture
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        move = chess.Move.from_uci("e4d5")
        mapping = self._get_dummy_mapping()
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 2
        # 1. Remove captured piece
        assert ops[0].target_square == "d5"
        assert ops[0].dest_square == "black_graveyard"
        assert ops[0].piece_name == "piece_d5"
        assert ops[0].is_capture is True
        # 2. Move attacking piece
        assert ops[1].target_square == "e4"
        assert ops[1].dest_square == "d5"
        assert ops[1].piece_name == "piece_e4"

    def test_en_passant_move_ops(self):
        """
        Verify that en passant generates a capture at the correct square (not the destination).
        """
        planner = OperationPlanner()
        # White pawn on f5, Black pawn just moved from e7 to e5
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3")
        move = chess.Move.from_uci("f5e6")
        mapping = self._get_dummy_mapping()
        assert board.is_en_passant(move)
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 2
        # 1. Remove captured piece (the black pawn on e5, NOT e6)
        assert ops[0].target_square == "e5"
        assert ops[0].dest_square == "black_graveyard"
        assert ops[0].piece_name == "piece_e5"
        assert ops[0].is_capture is True
        # 2. Move attacking piece (f5 -> e6)
        assert ops[1].target_square == "f5"
        assert ops[1].dest_square == "e6"
        assert ops[1].piece_name == "piece_f5"

    def test_pawn_promotion_move_ops(self):
        """
        Verify that pawn promotion generates an operation with a promotion attribute.
        """
        planner = OperationPlanner()
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        move = chess.Move.from_uci("a7a8q")
        mapping = self._get_dummy_mapping()
        assert move.promotion == chess.QUEEN
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 1
        assert ops[0].target_square == "a7"
        assert ops[0].dest_square == "a8"
        assert ops[0].piece_name == "piece_a7"
        assert ops[0].promotion == chess.QUEEN

    def test_pawn_promotion_capture_move_ops(self):
        """
        Verify that a capturing promotion generates both a capture and a promotion operation.
        """
        planner = OperationPlanner()
        board = chess.Board("1r6/P7/8/8/8/8/8/k6K w - - 0 1")
        move = chess.Move.from_uci("a7b8q")
        mapping = self._get_dummy_mapping()
        assert move.promotion == chess.QUEEN
        assert board.is_capture(move)
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 2
        # 1. Remove captured piece (black rook on b8)
        assert ops[0].target_square == "b8"
        assert ops[0].dest_square == "black_graveyard"
        assert ops[0].piece_name == "piece_b8"
        assert ops[0].is_capture is True
        # 2. Move attacking piece (a7 -> b8)
        assert ops[1].target_square == "a7"
        assert ops[1].dest_square == "b8"
        assert ops[1].piece_name == "piece_a7"
        assert ops[1].promotion == chess.QUEEN

    def test_castling_move_ops(self):
        """
        Verify that kingside castling generates two operations: king then rook.
        """
        planner = OperationPlanner()
        # White king on e1, Rook on h1, no pieces in between
        board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        move = chess.Move.from_uci("e1g1")
        mapping = self._get_dummy_mapping()
        assert board.is_castling(move)
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 2
        # 1. Move king (e1 -> g1)
        assert ops[0].target_square == "e1"
        assert ops[0].dest_square == "g1"
        assert ops[0].piece_name == "piece_e1"
        # 2. Move rook (h1 -> f1)
        assert ops[1].target_square == "h1"
        assert ops[1].dest_square == "f1"
        assert ops[1].piece_name == "piece_h1"

    def test_queenside_castling_move_ops(self):
        """
        Verify that queenside castling generates correct king and rook moves.
        """
        planner = OperationPlanner()
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1c1")
        mapping = self._get_dummy_mapping()
        assert board.is_castling(move)
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 2
        assert ops[0].target_square == "e1"
        assert ops[0].dest_square == "c1"
        assert ops[1].target_square == "a1"
        assert ops[1].dest_square == "d1"

    def test_pawn_under_promotion_to_knight(self):
        """
        Verify that under-promotion (e.g., to a knight) is correctly recorded.
        """
        planner = OperationPlanner()
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        move = chess.Move.from_uci("a7a8n")
        mapping = self._get_dummy_mapping()
        assert move.promotion == chess.KNIGHT
        
        ops = planner.generate_operations(move, board, mapping)
        
        assert len(ops) == 1
        assert ops[0].promotion == chess.KNIGHT
