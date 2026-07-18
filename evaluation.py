import chess
import os
from positions import EG_MAP, MG_MAP
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np
import math
from enum import Enum, auto
import bulletchess as bc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
model = load_model("chess.keras")
PHASE_WEIGHTS = {
    bc.PAWN: 0,
    bc.KNIGHT: 1,
    bc.BISHOP: 1,
    bc.ROOK: 2,
    bc.QUEEN: 4,
    bc.KING: 0
}

TOTAL_PHASE =  24

class ScoreType(Enum):
    CENTIPAWNS = auto()
    NORMALIZED = auto()

class Handcrafted:
    scope_type = ScoreType.CENTIPAWNS
    window_margin = 70
    queen = 900
    def __call__(self, board: bc.Board):
            piece_values = {
                bc.PAWN: 100,
                bc.ROOK: 500,
                bc.KNIGHT: 320,
                bc.BISHOP: 330,
                bc.QUEEN: 900,
                bc.KING: 2000,
            }
            score = 0
            mg_score = 0
            eg_score = 0
            phase = self.game_phase(board)
    
            for square in bc.SQUARES:
                piece = board[square]
                if piece is None:
                    continue
    
                value = piece_values[piece.piece_type]
    
                if piece.color == bc.WHITE:
                    score += value
                    mg_score += MG_MAP[piece.piece_type][square.index()]
                    eg_score += EG_MAP[piece.piece_type][square.index()]
                else:
                    mirrored_index = square.index() ^ 56  # vertical mirror, same trick as chess.square_mirror
                    score -= value
                    mg_score -= MG_MAP[piece.piece_type][mirrored_index]
                    eg_score -= EG_MAP[piece.piece_type][mirrored_index]
    
            score += (mg_score * (24 - phase) + eg_score * phase) // 24
            score = score if board.turn == bc.WHITE else -score
    
            if phase >= 20:
                enemy_color = board.turn.opposite
                enemy_king_square = next(iter(board[enemy_color, bc.KING]))
    
                file = enemy_king_square.index() % 8
                rank = enemy_king_square.index() // 8
    
                distance_from_center = abs(file - 3.5) + abs(rank - 3.5)
                score += distance_from_center * 20
    
            return score
    
    def game_phase(self, board: bc.Board):
            phase = TOTAL_PHASE
            for piece_type, weight in PHASE_WEIGHTS.items():
                phase -= weight * (
                    len(board[bc.WHITE, piece_type]) +
                    len(board[bc.BLACK, piece_type])
                )
            return phase / TOTAL_PHASE


class NNEvaluation():
    scope_type = ScoreType.NORMALIZED
    window_margin = .05
    queen = .98
    MAX_PIECES = 32
    PIECE_TYPE_TO_IDX = {
        bc.PAWN: 0,
        bc.KNIGHT: 1,
        bc.BISHOP: 2,
        bc.ROOK: 3,
        bc.QUEEN: 4,
    }

    def __init__(self, model=model) -> None:
        self.model = model

    def __call__(self, board: bc.Board):
        us_idx, them_idx = self.board_to_halfkp(board)
        us_idx = self.pad_indices(us_idx)[None, :]
        them_idx = self.pad_indices(them_idx)[None, :]
        out = float(self.model({"us_idx": us_idx, "them_idx": them_idx}, training=False)[0, 0])
        return out

    def mirror_sq(self, square: int) -> int:
        return square ^ 56

    def orient(self, square: int, perspective_is_white: bool) -> int:
        return square if perspective_is_white else self.mirror_sq(square)

    def halfkp_from_board(self, board: bc.Board, king_sq_white: int, king_sq_black: int, perspective_is_white: bool):
        king_sq = king_sq_white if perspective_is_white else king_sq_black
        k = self.orient(king_sq, perspective_is_white)
        indices = []
        for square in bc.SQUARES:
            piece = board[square]
            if piece is None or piece.piece_type == bc.KING:
                continue
            ptype = self.PIECE_TYPE_TO_IDX[piece.piece_type]
            is_white = piece.color == bc.WHITE
            relative = 0 if is_white == perspective_is_white else 1
            p_idx = relative * 5 + ptype
            sq = self.orient(square.index(), perspective_is_white)
            indices.append(k * 640 + p_idx * 64 + sq)
        return indices

    def board_to_halfkp(self, board: bc.Board):
        us_is_white = board.turn == bc.WHITE
        king_sq_white = next(iter(board[bc.WHITE, bc.KING])).index()
        king_sq_black = next(iter(board[bc.BLACK, bc.KING])).index()
        us_idx = self.halfkp_from_board(board, king_sq_white, king_sq_black, us_is_white)
        them_idx = self.halfkp_from_board(board, king_sq_white, king_sq_black, not us_is_white)
        return np.array(us_idx, dtype=np.int32), np.array(them_idx, dtype=np.int32)

    def pad_indices(self, idx):
        padded = np.full(self.MAX_PIECES, -1, dtype=np.int32)
        padded[:len(idx)] = idx
        return padded
