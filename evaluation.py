from typing import override
import random
import os
from positions import EG_MAP, MG_MAP
from tensorflow.keras.models import load_model
import numpy as np
from enum import Enum, auto
import bulletchess as bc
from numba import njit

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
    def set_board(self, board):
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
    @override
    def set_board(self, board):
        self.board = board
        self.score_builded = False
    piece_values = {
        bc.PAWN: 100,
        bc.ROOK: 500,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.QUEEN: 900,
        bc.KING: 2000,
    }
 
    def __init__(self, board: bc.Board = None) -> None:
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

@njit(cache=True)
def _numba_forward(acc, half_dim, is_white_turn, w0, b0, w1, b1, w2, b2, w3, b3):
    if is_white_turn:
        x = np.empty(2 * half_dim, dtype=np.float32)
        x[:half_dim] = acc[:half_dim]
        x[half_dim:] = acc[half_dim:]
    else:
        x = np.empty(2 * half_dim, dtype=np.float32)
        x[:half_dim] = acc[half_dim:]
        x[half_dim:] = acc[:half_dim]
        
    x = np.dot(x, w0) + b0
    x = np.maximum(x, 0)
    x = np.dot(x, w1) + b1
    x = np.maximum(x, 0)
    x = np.dot(x, w2) + b2
    x = np.maximum(x, 0)
    x = np.dot(x, w3) + b3
    x = np.tanh(x)
    return x[0]

@njit(cache=True)
def _numba_do(acc, emb, king_sq_w, king_sq_b, origin_idx, dest_idx, 
              moving_type, moving_color, dest_type, 
              is_capture, captured_type, captured_color, captured_sq, 
              is_castle, r_from, r_to, king_moved, piece_type_to_idx):
    
    half_dim = emb.shape[1]
    
    # --- White Perspective ---
    if not (king_moved and moving_color == 0):
        kw = king_sq_w
        if is_capture:
            is_own = (captured_color == 0)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[captured_type]
            # ADDED + 1 TO MATCH KERAS MODEL SHIFT
            idx = kw * 640 + p_idx * 64 + captured_sq + 1
            acc[:half_dim] -= emb[idx]
        
        if not king_moved:
            is_own = (moving_color == 0)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[moving_type]
            idx = kw * 640 + p_idx * 64 + origin_idx + 1
            acc[:half_dim] -= emb[idx]
            
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[dest_type]
            idx = kw * 640 + p_idx * 64 + dest_idx + 1
            acc[:half_dim] += emb[idx]
            
        elif is_castle:
            is_own = (moving_color == 0)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[3] # ROOK = 3
            idx = kw * 640 + p_idx * 64 + r_from + 1
            acc[:half_dim] -= emb[idx]
            idx = kw * 640 + p_idx * 64 + r_to + 1
            acc[:half_dim] += emb[idx]

    # --- Black Perspective ---
    if not (king_moved and moving_color == 1):
        kb_mir = king_sq_b ^ 56
        if is_capture:
            sq = captured_sq ^ 56
            is_own = (captured_color == 1)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[captured_type]
            idx = kb_mir * 640 + p_idx * 64 + sq + 1
            acc[half_dim:] -= emb[idx]
        
        if not king_moved:
            sq = origin_idx ^ 56
            is_own = (moving_color == 1)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[moving_type]
            idx = kb_mir * 640 + p_idx * 64 + sq + 1
            acc[half_dim:] -= emb[idx]
            
            sq = dest_idx ^ 56
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[dest_type]
            idx = kb_mir * 640 + p_idx * 64 + sq + 1
            acc[half_dim:] += emb[idx]
            
        elif is_castle:
            is_own = (moving_color == 1)
            p_idx = 0 if is_own else 5
            p_idx += piece_type_to_idx[3] # ROOK = 3
            idx = kb_mir * 640 + p_idx * 64 + (r_from ^ 56) + 1
            acc[half_dim:] -= emb[idx]
            idx = kb_mir * 640 + p_idx * 64 + (r_to ^ 56) + 1
            acc[half_dim:] += emb[idx]
            
    if king_moved:
        if moving_color == 0:
            return dest_idx, king_sq_b
        else:
            return king_sq_w, dest_idx
    return king_sq_w, king_sq_b

class NNEvaluation(Evaluation):
    @property
    @override
    def scope_type(self):
        return ScoreType.CENTIPAWNS
    @override
    def set_board(self, board):
        self.board = board
        self.score_builded = False
    @property
    @override
    def window_margin(self):
        return 70

    @property
    @override
    def queen(self):
        return 900

    MAX_PIECES = 32
    PIECE_VALUES = {
        bc.PAWN: 100,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.ROOK: 500,
        bc.QUEEN: 900,
    }
    PIECE_VALUES_INT = [100, 320, 330, 500, 900, 0]

    def __init__(self, board: bc.Board = None) -> None:
        self.board = board
        
        keras_model = load_model("chess.keras")
        self.embedding = np.ascontiguousarray(keras_model.get_layer("embedding").get_weights()[0], dtype=np.float32)

        self.score_builded = False
        self.half_dim = self.embedding.shape[1]
        
        self.accumulator = np.zeros(2 * self.half_dim, dtype=np.float32)
        self.material_score = 0  
        
        self.king_sq_white = None
        self.king_sq_black = None

        self.PIECE_TYPE_TO_INT = {
            bc.PAWN: 0, bc.KNIGHT: 1, bc.BISHOP: 2, bc.ROOK: 3, bc.QUEEN: 4, bc.KING: 5
        }
        self.COLOR_TO_INT = {bc.WHITE: 0, bc.BLACK: 1}
        
        self.PIECE_TYPE_TO_IDX_NP = np.array([0, 1, 2, 3, 4, 0], dtype=np.int32)

        w0, b0 = keras_model.get_layer("dense").get_weights()
        w1, b1 = keras_model.get_layer("dense_1").get_weights()
        w2, b2 = keras_model.get_layer("dense_2").get_weights()
        w3, b3 = keras_model.get_layer("dense_3").get_weights()
        
        self.w0 = np.ascontiguousarray(w0, dtype=np.float32)
        self.b0 = np.ascontiguousarray(b0, dtype=np.float32)
        self.w1 = np.ascontiguousarray(w1, dtype=np.float32)
        self.b1 = np.ascontiguousarray(b1, dtype=np.float32)
        self.w2 = np.ascontiguousarray(w2, dtype=np.float32)
        self.b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self.w3 = np.ascontiguousarray(w3, dtype=np.float32)
        self.b3 = np.ascontiguousarray(b3, dtype=np.float32)

        self.moves = []

    def _get_piece_int(self, piece_type):
        return self.PIECE_TYPE_TO_INT[piece_type]
        
    def _get_color_int(self, color):
        return self.COLOR_TO_INT[color]

    def __call__(self):
        if not self.score_builded:
            self.build()
        return self._model_call()

    def _model_call(self):
        raw_output = float(_numba_forward(
            self.accumulator, self.half_dim, self.board.turn == bc.WHITE,
            self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
        ))
        raw_output = max(-1 + 1e-4, min(1 - 1e-4, raw_output))
        nn_cp = np.arctanh(raw_output) * 400.0
        if self.board.turn == bc.WHITE:
            return nn_cp + 4 * self.material_score

        else:
            return nn_cp - 4 * self.material_score

    def _build_half(self, perspective_is_white, king_sq):
        k = king_sq if perspective_is_white else king_sq ^ 56
        indices = []
        
        for color in (bc.WHITE, bc.BLACK):
            is_own = (color == bc.WHITE) if perspective_is_white else (color == bc.BLACK)
            rel_offset = 0 if is_own else 5
            for piece_type, p in self.PIECE_TYPE_TO_INT.items():
                if piece_type == bc.KING: continue
                p_idx = rel_offset + p
                base = k * 640 + p_idx * 64
                for square in self.board[color, piece_type]:
                    sq = square.index()
                    sq = sq if perspective_is_white else sq ^ 56
                    # ADDED + 1 TO MATCH KERAS MODEL SHIFT
                    indices.append(base + sq + 1)
                    
        half_start = 0 if perspective_is_white else self.half_dim
        half_end = self.half_dim if perspective_is_white else 2 * self.half_dim
        
        if indices:
            idx_arr = np.array(indices, dtype=np.intp)
            np.add.reduce(self.embedding[idx_arr], axis=0, out=self.accumulator[half_start:half_end])
        else:
            self.accumulator[half_start:half_end].fill(0)

    def build(self):
        self.king_sq_white = next(iter(self.board[bc.WHITE, bc.KING])).index()
        self.king_sq_black = next(iter(self.board[bc.BLACK, bc.KING])).index()
        self.accumulator.fill(0)
        self._build_half(True, self.king_sq_white)
        self._build_half(False, self.king_sq_black)
        self.material_score = 0
        for piece_type, val in self.PIECE_VALUES.items():
            self.material_score += len(self.board[bc.WHITE, piece_type]) * val
            self.material_score -= len(self.board[bc.BLACK, piece_type]) * val
            
        self.score_builded = True

    @override
    def do(self, move: bc.Move):
        if not self.score_builded:
            self.build()

        origin_idx = move.origin.index()
        dest_idx = move.destination.index()
        moving_piece = self.board[move.origin]
        
        moving_type_int = self._get_piece_int(moving_piece.piece_type)
        moving_color_int = self._get_color_int(moving_piece.color)
        king_moved = (moving_type_int == 5)

        is_capture = False
        captured_type_int = -1
        captured_color_int = -1
        captured_sq_idx = -1

        dest_content = self.board[move.destination]
        if dest_content is not None:
            is_capture = True
            captured_type_int = self._get_piece_int(dest_content.piece_type)
            captured_color_int = self._get_color_int(dest_content.color)
            captured_sq_idx = dest_idx
        elif moving_type_int == 0 and (dest_idx % 8) != (origin_idx % 8):
            is_capture = True
            captured_sq_idx = (origin_idx // 8) * 8 + (dest_idx % 8)
            captured_piece = self.board[bc.SQUARES[captured_sq_idx]]
            captured_type_int = self._get_piece_int(captured_piece.piece_type)
            captured_color_int = self._get_color_int(captured_piece.color)

        is_castle = False
        r_from = -1
        r_to = -1
        if king_moved:
            file_diff = (dest_idx % 8) - (origin_idx % 8)
            if file_diff == 2:
                is_castle = True
                r_from = dest_idx + 1
                r_to = dest_idx - 1
            elif file_diff == -2:
                is_castle = True
                r_from = dest_idx - 2
                r_to = dest_idx + 1

        self.moves.append((self.accumulator.copy(), self.king_sq_white, self.king_sq_black, self.material_score))

        promo = move.promotion
        dest_type_int = self._get_piece_int(promo) if promo is not None else moving_type_int

        if is_capture:
            victim_val = self.PIECE_VALUES_INT[captured_type_int]
            if captured_color_int == 0: 
                self.material_score -= victim_val
            else:
                self.material_score += victim_val
                
        if promo is not None:
            promo_val = self.PIECE_VALUES_INT[dest_type_int]
            if moving_color_int == 0:   
                self.material_score += promo_val - self.PIECE_VALUES_INT[0]
            else:                        
                self.material_score -= promo_val - self.PIECE_VALUES_INT[0]

        self.king_sq_white, self.king_sq_black = _numba_do(
            self.accumulator, self.embedding, self.king_sq_white, self.king_sq_black,
            origin_idx, dest_idx, moving_type_int, moving_color_int, dest_type_int,
            is_capture, captured_type_int, captured_color_int, captured_sq_idx,
            is_castle, r_from, r_to, king_moved, self.PIECE_TYPE_TO_IDX_NP
        )

        if king_moved:
            self.board.apply(move)
            self._build_half(moving_color_int == 0, dest_idx)
            self.board.undo()

    @override
    def undo(self):
        if not self.moves:
            raise IndexError("nothing to undo")
        acc, self.king_sq_white, self.king_sq_black, self.material_score = self.moves.pop()
        self.accumulator[:] = acc        


        
def run_validation_test():
    print("Starting rigorous validation test...")
    
    board = bc.Board()
    inc = NNEvaluation(board)
    full = NNEvaluation(board)
    
    print("Warming up Numba JIT...")
    _ = inc()
    _ = full()
    
    move_history = []
    
    for i in range(10000):
        legal_moves = board.legal_moves()
        if not legal_moves:
            print(f"Game ended at move {i}. Resetting board.")
            board = bc.Board()
            inc = NNEvaluation(board)
            full = NNEvaluation(board)
            move_history = []
            continue
            
        move = random.choice(legal_moves)
        uci_move = str(move)
        move_history.append(uci_move)
        
        # Extract move info BEFORE applying to the board
        moving_piece = board[move.origin]
        is_cap = move.is_capture(board)
        moving_type = moving_piece.piece_type if moving_piece else None
        moving_color = moving_piece.color if moving_piece else None
        
        # --- 1. PRE-MOVE VERIFICATION ---
        full.board = board.copy()
        full.score_builded = False
        full.build()
        
        if not np.allclose(inc.accumulator, full.accumulator, atol=1e-5):
            print("\n!!! PRE-MOVE MISMATCH DETECTED !!!")
            print("(This usually means the previous undo() was broken)")
            print(f"Move {i+1}: {uci_move}")
            print("Move History:", " ".join(move_history))
            print("FEN:", board.fen())
            diff = inc.accumulator - full.accumulator
            idx = np.argmax(np.abs(diff))
            print(f"Max Diff: {diff[idx]} at index {idx}")
            if idx < inc.half_dim:
                print("Bug is in WHITE perspective half.")
            else:
                print("Bug is in BLACK perspective half.")
            return

        # Save state before do() to test undo later
        prev_acc = inc.accumulator.copy()
        
        # --- APPLY MOVE ---
        inc.do(move)
        board.apply(move)
        
        # --- 2. POST-MOVE VERIFICATION ---
        full.board = board.copy()
        full.score_builded = False
        full.build()
        
        if not np.allclose(inc.accumulator, full.accumulator, atol=1e-7):
            print("\n!!! BUG DETECTED IN do() !!!")
            print(f"Move {i+1}: {uci_move}")
            print("Move History:", " ".join(move_history))
            print("FEN BEFORE move:", board.fen()) # Board is already applied here, so this is the post-move FEN
            print(f"Moving Piece: {moving_color} {moving_type}")
            print(f"Is Capture: {is_cap}")
            print(f"Promotion: {move.promotion}")
            
            diff = inc.accumulator - full.accumulator
            idx = np.argmax(np.abs(diff))
            print(f"\nMax Difference: {diff[idx]}")
            print(f"Index of Max Diff: {idx}")
            
            if idx < inc.half_dim:
                print("Bug is in WHITE perspective half!")
            else:
                print("Bug is in BLACK perspective half!")
            return
            
        # --- 3. UNDO VERIFICATION ---
        board.undo()
        inc.undo()
        
        if not np.allclose(inc.accumulator, prev_acc, atol=1e-5):
            print("\n!!! BUG DETECTED IN undo() !!!")
            print(f"Move {i+1}: {uci_move}")
            print("Move History:", " ".join(move_history))
            print(f"Moving Piece: {moving_color} {moving_type}")
            print(f"Is Capture: {is_cap}")
            diff = inc.accumulator - prev_acc
            idx = np.argmax(np.abs(diff))
            print(f"Max Diff: {diff[idx]} at index {idx}")
            if idx < inc.half_dim:
                print("Bug is in WHITE perspective half.")
            else:
                print("Bug is in BLACK perspective half.")
            return
        print(f"Move {i+1}: {uci_move}")

    print("\nTest passed successfully for 10000 moves!")

if __name__ == "__main__":
    run_validation_test()