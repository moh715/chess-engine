from typing import override

import os
from positions import EG_MAP, MG_MAP
from tensorflow.keras.models import load_model
from tensorflow import function
from tensorflow import expand_dims
import numpy as np
import math
from enum import Enum, auto
import bulletchess as bc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

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

from abc import ABC, abstractmethod

class Evaluation(ABC):
    @property
    @abstractmethod
    def scope_type(self):
        ...

    @property
    @abstractmethod
    def window_margin(self):
        ...
    @property
    @abstractmethod
    def queen(self):
        ...

    @abstractmethod
    def __call__(self):
        ...
    @abstractmethod
    def do(self, move):
        ...
    @abstractmethod
    def undo(self):
        ...


class Handcrafted(Evaluation):
    @property
    @override
    def scope_type(self):
        return ScoreType.CENTIPAWNS
    @property
    @override
    def window_margin(self):
        return 70
    @property
    @override
    def queen(self):
        return 900
    piece_values = {
        bc.PAWN: 100,
        bc.ROOK: 500,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.QUEEN: 900,
        bc.KING: 2000,
    }
 
    def __init__(self, board: bc.Board) -> None:
        self.score = 0
        self.moves = []
        self.board = board
        self.score_builded = False
        self.piece_score = 0
        self.mg_score = 0
        self.eg_score = 0
        self.phase = 0
 
    def __call__(self):
        if not self.score_builded:
            self.build()
 
        score = (self.mg_score * (24 - self.phase) + self.eg_score * self.phase) // 24
        score += self.piece_score
 
        if self.phase <= 4:
            if self.piece_score > 0:
                advantaged_color = bc.WHITE
            elif self.piece_score < 0:
                advantaged_color = bc.BLACK
            else:
                advantaged_color = None
 
            if advantaged_color is not None:
                weaker_king_square = next(iter(self.board[advantaged_color.opposite, bc.KING]))
                file = weaker_king_square.index() % 8
                rank = weaker_king_square.index() // 8
                distance_from_center = abs(file - 3.5) + abs(rank - 3.5)
                bonus = distance_from_center * 20
                score += bonus if advantaged_color == bc.WHITE else -bonus
 
        score = score if self.board.turn == bc.WHITE else -score
        return score
 
    def do(self, move: bc.Move):
        if not self.score_builded:
            self.build()
        if move not in self.board.legal_moves():
            raise ValueError("move is Ilegal")
 
        self.moves.append((self.phase, self.piece_score, self.mg_score, self.eg_score))
        piece = self.board[move.origin]
        origin_color = 1 if piece.color == bc.WHITE else -1
 
        if move.is_capture(self.board):
            captured_piece = self.board[move.destination]
            captured_square_index = move.destination.index()
 
            if captured_piece is None:
                captured_square_index = move.destination.index() % 8 + (move.origin.index() // 8) * 8
                captured_piece = self.board[bc.SQUARES[captured_square_index]]
 
            value = self.piece_values[captured_piece.piece_type]
            value = -value if captured_piece.color == bc.WHITE else value
            self.piece_score += value
            self.phase -= PHASE_WEIGHTS[captured_piece.piece_type]
 
            if captured_piece.color == bc.WHITE:
                self.mg_score -= MG_MAP[captured_piece.piece_type][captured_square_index]
                self.eg_score -= EG_MAP[captured_piece.piece_type][captured_square_index]
            else:
                mirrored_index = captured_square_index ^ 56
                self.mg_score += MG_MAP[captured_piece.piece_type][mirrored_index]
                self.eg_score += EG_MAP[captured_piece.piece_type][mirrored_index]
 
        if move.promotion:
            self.piece_score += (self.piece_values[move.promotion] - self.piece_values[bc.PAWN]) * origin_color
            self.phase -= (PHASE_WEIGHTS[move.promotion] - PHASE_WEIGHTS[bc.PAWN])
 
            if piece.color == bc.WHITE:
                self.mg_score -= MG_MAP[bc.PAWN][move.origin.index()]
                self.eg_score -= EG_MAP[bc.PAWN][move.origin.index()]
                self.mg_score += MG_MAP[move.promotion][move.destination.index()]
                self.eg_score += EG_MAP[move.promotion][move.destination.index()]
            else:
                origin_mirrored = move.origin.index() ^ 56
                dest_mirrored = move.destination.index() ^ 56
                self.mg_score += MG_MAP[bc.PAWN][origin_mirrored]
                self.eg_score += EG_MAP[bc.PAWN][origin_mirrored]
                self.mg_score -= MG_MAP[move.promotion][dest_mirrored]
                self.eg_score -= EG_MAP[move.promotion][dest_mirrored]
 
        if move.is_castling(self.board):
            if piece.color == bc.WHITE:
                if move.destination.index() == 6:
                    self._move_pst(bc.ROOK, bc.WHITE, 7, 5)
                else:
                    self._move_pst(bc.ROOK, bc.WHITE, 0, 3)
            else:
                if move.destination.index() == 62:
                    self._move_pst(bc.ROOK, bc.BLACK, 63, 61)
                else:
                    self._move_pst(bc.ROOK, bc.BLACK, 56, 59)
 
        if not move.promotion:
            self._move_pst(piece.piece_type, piece.color, move.origin.index(), move.destination.index())
        
 
    def _move_pst(self, piece_type, color, from_sq, to_sq):
        if color == bc.WHITE:
            self.mg_score -= MG_MAP[piece_type][from_sq]
            self.eg_score -= EG_MAP[piece_type][from_sq]
            self.mg_score += MG_MAP[piece_type][to_sq]
            self.eg_score += EG_MAP[piece_type][to_sq]
        else:
            from_sq ^= 56
            to_sq ^= 56
            self.mg_score += MG_MAP[piece_type][from_sq]
            self.eg_score += EG_MAP[piece_type][from_sq]
            self.mg_score -= MG_MAP[piece_type][to_sq]
            self.eg_score -= EG_MAP[piece_type][to_sq]
 
    def undo(self):
        self.phase, self.piece_score, self.mg_score, self.eg_score = self.moves.pop()
 
    def build(self):
        score = 0
        mg_score = 0
        eg_score = 0
        phase = self.game_phase()
 
        for square in bc.SQUARES:
            piece = self.board[square]
            if piece is None:
                continue
 
            value = self.piece_values[piece.piece_type]
 
            if piece.color == bc.WHITE:
                score += value
                mg_score += MG_MAP[piece.piece_type][square.index()]
                eg_score += EG_MAP[piece.piece_type][square.index()]
            else:
                mirrored_index = square.index() ^ 56
                score -= value
                mg_score -= MG_MAP[piece.piece_type][mirrored_index]
                eg_score -= EG_MAP[piece.piece_type][mirrored_index]
 
        self.piece_score = score
        self.mg_score = mg_score
        self.eg_score = eg_score
        self.phase = phase
        self.score_builded = True
 
    def game_phase(self):
        phase = TOTAL_PHASE
        for piece_type, weight in PHASE_WEIGHTS.items():
            phase -= weight * (
                len(self.board[bc.WHITE, piece_type]) +
                len(self.board[bc.BLACK, piece_type])
            )
        return phase

class NNEvaluation(Evaluation):
    @property
    @override
    def scope_type(self):
        return ScoreType.NORMALIZED

    @property
    @override
    def window_margin(self):
        return 0.05

    @property
    @override
    def queen(self):
        return 0.98
    MAX_PIECES = 32
    PIECE_TYPE_TO_IDX = {
        bc.PAWN: 0,
        bc.KNIGHT: 1,
        bc.BISHOP: 2,
        bc.ROOK: 3,
        bc.QUEEN: 4,
    }

    def __init__(self, board:bc.Board, model=None) -> None:
        self.model = model
        self.board = board
        if not model:
            self.model = load_model("chess.keras")
        self.embedding = self.model.get_layer("embedding").get_weights()[0]
        
        self.weights = []
        self.biases = []
        
        for i in range(5):
            W, b = self.model.get_layer(f"dense{'' if i == 0 else '_' + str(i)}").get_weights()
            self.weights.append(W)
            self.biases.append(b)
    def __call__(self):
        us_idx, them_idx = self.board_to_halfkp()
        # us_idx = self.pad_indices(us_idx)[None, :]
        # them_idx = self.pad_indices(them_idx)[None, :]
        return self._model_call(us_idx, them_idx)
        
    def do(self, move:bc.Move):
        pass

    @override
    def undo(self):
        pass


        
    def _model_call(self, us_idx, them_idx):
        acc_us = self.embedding[us_idx].sum(axis=0)
        acc_them = self.embedding[them_idx].sum(axis=0)
        x = np.concatenate([acc_us, acc_them])
        
        for W, b in zip(self.weights, self.biases):
            x = np.maximum(x @ W + b, 0)
        return x

        
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

    def board_to_halfkp(self):
        us_is_white = self.board.turn == bc.WHITE
        king_sq_white = next(iter(self.board[bc.WHITE, bc.KING])).index()
        king_sq_black = next(iter(self.board[bc.BLACK, bc.KING])).index()
        us_idx = self.halfkp_from_board(self.board, king_sq_white, king_sq_black, us_is_white)
        them_idx = self.halfkp_from_board(self.board, king_sq_white, king_sq_black, not us_is_white)
        return np.array(us_idx, dtype=np.int32), np.array(them_idx, dtype=np.int32)

    def pad_indices(self, idx):
        padded = np.full(self.MAX_PIECES, -1, dtype=np.int32)
        padded[:len(idx)] = idx
        return padded
