import chess
import os
from positions import EG_MAP, MG_MAP
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np
import math
from enum import Enum, auto

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
model = load_model("chess.keras")
PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0
}

TOTAL_PHASE =  24

class ScoreType(Enum):
    CENTIPAWNS = auto()
    NORMALIZED = auto()

class Handcrafted:
    scope_type = ScoreType.CENTIPAWNS
    window_margin = 70
    queen = 900
    def __call__(self, board: chess.Board):
        piece_values = {
            chess.PAWN: 100,
            chess.ROOK: 500,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.QUEEN: 900,
            chess.KING: 2000,
        }
        score = 0
        
        mg_score = 0
        eg_score = 0 
        phase = self.game_phase(board)
        for square, piece in board.piece_map().items():
            value = piece_values[piece.piece_type]
    
            if piece.color == chess.WHITE:
                score += value
                mg_score += MG_MAP[piece.piece_type][square]
                eg_score += EG_MAP[piece.piece_type][square]
            else:
                score -= value
                mg_score -= MG_MAP[piece.piece_type][chess.square_mirror(square)]
                eg_score -= EG_MAP[piece.piece_type][chess.square_mirror(square)]
        score += (mg_score * (24 - phase) + eg_score * phase) // 24
        score = score if board.turn == chess.WHITE else -score
        if phase >= 20:
            enemy_king = board.king(not board.turn)
            
            file = chess.square_file(enemy_king)
            rank = chess.square_rank(enemy_king)
            
            distance_from_center = abs(file - 3.5) + abs(rank - 3.5)
            
            score += distance_from_center * 20
        return score
    
    def game_phase(self, board:chess.Board):
        
        phase = TOTAL_PHASE
    
        for piece_type, weight in PHASE_WEIGHTS.items():
            phase -= weight * (
                len(board.pieces(piece_type, chess.WHITE)) +
                len(board.pieces(piece_type, chess.BLACK))
            )
    
        return phase / TOTAL_PHASE


class NNEvaluation():
    scope_type = ScoreType.NORMALIZED
    window_margin = .05
    queen = .98
    MAX_PIECES = 32

    PIECE_TYPE_TO_IDX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }

    def __init__(self, model=model) -> None:
        self.model = model

    def __call__(self, board: chess.Board):
        us_idx, them_idx = self.board_to_halfkp(board)
        us_idx = self.pad_indices(us_idx)[None, :]
        them_idx = self.pad_indices(them_idx)[None, :]
        out = float(self.model({"us_idx": us_idx, "them_idx": them_idx}, training=False)[0, 0])
        return out

    def mirror_sq(self, square):
        return square ^ 56

    def orient(self, square, perspective_is_white):
        return square if perspective_is_white else self.mirror_sq(square)

    def halfkp_from_board(self, board: chess.Board, king_sq_white, king_sq_black, perspective_is_white):
        king_sq = king_sq_white if perspective_is_white else king_sq_black
        k = self.orient(king_sq, perspective_is_white)

        indices = []
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            ptype = self.PIECE_TYPE_TO_IDX[piece.piece_type]
            is_white = piece.color  # chess.WHITE == True
            relative = 0 if is_white == perspective_is_white else 1
            p_idx = relative * 5 + ptype
            sq = self.orient(square, perspective_is_white)
            indices.append(k * 640 + p_idx * 64 + sq)
        return indices

    def board_to_halfkp(self, board: chess.Board):
        us_is_white = board.turn
        king_sq_white = board.king(chess.WHITE)
        king_sq_black = board.king(chess.BLACK)

        us_idx = self.halfkp_from_board(board, king_sq_white, king_sq_black, us_is_white)
        them_idx = self.halfkp_from_board(board, king_sq_white, king_sq_black, not us_is_white)
        return np.array(us_idx, dtype=np.int32), np.array(them_idx, dtype=np.int32)

    def pad_indices(self, idx):
        padded = np.full(self.MAX_PIECES, -1, dtype=np.int32)
        padded[:len(idx)] = idx
        return padded

