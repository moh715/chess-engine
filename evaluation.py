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
        if self.board[move.origin] is None:
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

    def __init__(self, board: bc.Board, model=None) -> None:
        self.model = model
        self.board = board
        if not model:
            self.model = load_model("chess.keras")
        self.embedding = self.model.get_layer("embedding").get_weights()[0]

        self.weights = []
        self.biases = []
        self.score_builded = False
        self.accumulator = 0

        self.moves = []

        self.half_dim = self.embedding.shape[1]
        self._white_slice = slice(0, self.half_dim)
        self._black_slice = slice(self.half_dim, 2 * self.half_dim)

        self.king_sq_white = None
        self.king_sq_black = None

        self._castle_rook_squares = {
            bc.WHITE_KINGSIDE:  (bc.H1.index(), bc.F1.index()),
            bc.WHITE_QUEENSIDE: (bc.A1.index(), bc.D1.index()),
            bc.BLACK_KINGSIDE:  (bc.H8.index(), bc.F8.index()),
            bc.BLACK_QUEENSIDE: (bc.A8.index(), bc.D8.index()),
        }

        for i in range(5):
            W, b = self.model.get_layer(f"dense{'' if i == 0 else '_' + str(i)}").get_weights()
            self.weights.append(W)
            self.biases.append(b)

    def __call__(self):
        if not self.score_builded:
            self.build()
        return self._model_call()

    def _model_call(self):
        x = self.accumulator
        for i in range(len(self.weights) - 1):
            x =np.maximum(x @ self.weights[i] + self.biases[i], 0)
        x = x @ self.weights[-1] + self.biases[-1]
        return x

    def mirror_sq(self, square: int) -> int:
        return square ^ 56

    def orient(self, square: int, perspective_is_white: bool) -> int:
        return square if perspective_is_white else self.mirror_sq(square)


    def _feature_index(self, perspective_is_white, king_sq, piece_type, piece_color, square_idx):
        k = self.orient(king_sq, perspective_is_white)
        relative = 0 if (piece_color == bc.WHITE) == perspective_is_white else 1
        p_idx = relative * 5 + self.PIECE_TYPE_TO_IDX[piece_type]
        sq = self.orient(square_idx, perspective_is_white)
        return k * 640 + p_idx * 64 + sq

    def _add(self, perspective_is_white, king_sq, piece_type, piece_color, square_idx):
        idx = self._feature_index(perspective_is_white, king_sq, piece_type, piece_color, square_idx)
        sl = self._white_slice if perspective_is_white else self._black_slice
        self.accumulator[sl] += self.embedding[idx]

    def _remove(self, perspective_is_white, king_sq, piece_type, piece_color, square_idx):
        idx = self._feature_index(perspective_is_white, king_sq, piece_type, piece_color, square_idx)
        sl = self._white_slice if perspective_is_white else self._black_slice
        self.accumulator[sl] -= self.embedding[idx]

    def _build_half(self, perspective_is_white, king_sq):
        """Recompute one perspective's half from scratch by scanning the board.
        Needed on the initial build(), and afterwards only for whichever half's
        own king just moved — that half's anchor square changed, so every one
        of its feature indices is stale and there's nothing worth patching."""
        k = self.orient(king_sq, perspective_is_white)
        indices = []
        for color in (bc.WHITE, bc.BLACK):
            relative = 0 if (color == bc.WHITE) == perspective_is_white else 1
            for piece_type, p in self.PIECE_TYPE_TO_IDX.items():
                p_idx = relative * 5 + p
                for square in self.board[color, piece_type]:  # C-backed bitboard iteration
                    sq = self.orient(square.index(), perspective_is_white)
                    indices.append(k * 640 + p_idx * 64 + sq)
        half = (self.embedding[indices].sum(axis=0) if indices
                else np.zeros(self.half_dim, dtype=self.embedding.dtype))
        sl = self._white_slice if perspective_is_white else self._black_slice
        self.accumulator[sl] = half

    def build(self):
        self.king_sq_white = next(iter(self.board[bc.WHITE, bc.KING])).index()
        self.king_sq_black = next(iter(self.board[bc.BLACK, bc.KING])).index()
        self.accumulator = np.zeros(2 * self.half_dim, dtype=self.embedding.dtype)
        self._build_half(True, self.king_sq_white)
        self._build_half(False, self.king_sq_black)
        self.score_builded = True

    # ---- do / undo ----------------------------------------------------------

    @override
    def do(self, move: bc.Move):
        if not self.score_builded:
            self.build()

        origin_idx = move.origin.index()
        dest_idx = move.destination.index()
        moving_piece = self.board[move.origin]
        assert moving_piece is not None, "do() called with an empty origin square"
        moving_type = moving_piece.piece_type
        moving_color = moving_piece.color
        king_moved = moving_type == bc.KING

        # What's being captured, if anything (read from the pre-move board).
        captured_piece = None
        captured_sq_idx = None
        if move.is_capture(self.board):
            dest_content = self.board[move.destination]
            if dest_content is None:
                # En passant: the captured pawn sits beside the destination, not on it.
                captured_sq_idx = (origin_idx // 8) * 8 + (dest_idx % 8)
                captured_piece = self.board[bc.SQUARES[captured_sq_idx]]
            else:
                captured_sq_idx = dest_idx
                captured_piece = dest_content

        # Castling drags the rook along too.
        castle_rook_squares = None
        if king_moved:
            ctype = move.castling_type(self.board)
            if ctype is not None:
                castle_rook_squares = self._castle_rook_squares[ctype]

        self.moves.append((self.accumulator.copy(), self.king_sq_white, self.king_sq_black))

        promo = move.promotion
        dest_type = promo if promo is not None else moving_type
        own_half_is_white = (moving_color == bc.WHITE) if king_moved else None

        for persp_is_white, king_sq in ((True, self.king_sq_white), (False, self.king_sq_black)):
            if king_moved and persp_is_white == own_half_is_white:
                # This half's own king just moved -- fully rebuilt below instead.
                continue

            if captured_piece is not None:
                self._remove(persp_is_white, king_sq, captured_piece.piece_type,
                              captured_piece.color, captured_sq_idx)

            if not king_moved:
                # Kings are never encoded as features, so a king move has nothing
                # to add/remove here -- only capture/rook-drag side effects matter.
                self._remove(persp_is_white, king_sq, moving_type, moving_color, origin_idx)
                self._add(persp_is_white, king_sq, dest_type, moving_color, dest_idx)

            if castle_rook_squares is not None:
                r_from, r_to = castle_rook_squares
                self._remove(persp_is_white, king_sq, bc.ROOK, moving_color, r_from)
                self._add(persp_is_white, king_sq, bc.ROOK, moving_color, r_to)

        if king_moved:
            self.board.apply(move)
            self._build_half(own_half_is_white, dest_idx)
            self.board.undo()
            if own_half_is_white:
                self.king_sq_white = dest_idx
            else:
                self.king_sq_black = dest_idx

    @override
    def undo(self):
        if not self.moves:
            raise IndexError("nothing to undo")
        self.accumulator, self.king_sq_white, self.king_sq_black = self.moves.pop()
    


if __name__ == "__main__":
    board = bc.Board()
    inc = NNEvaluation(board)
    full = NNEvaluation(board)
    for i, move in enumerate(board.legal_moves()):
    
        before = inc()
    
        inc.do(move)
        board.apply(move)
    
        full.build()          # rebuild from scratch
    
        if not np.allclose(inc.accumulator, full.accumulator):
            print(i)
            diff = inc.accumulator - full.accumulator
            print(np.max(np.abs(inc.accumulator - full.accumulator)))
            idx = np.argmax(np.abs(diff))
            print(idx, diff[idx])
            print(move)
        assert np.max(np.abs(inc.accumulator - full.accumulator)) < 1e-6
    
        board.undo()
        inc.undo()
    
        assert abs(inc() - before) < 1e-6