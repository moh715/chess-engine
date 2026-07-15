import chess
import os
from positions import EG_MAP, MG_MAP
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np
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



def evaluate(board: chess.Board):
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
    phase = game_phase(board)
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
    # if board.is_repetition():
    #         score += -200
    if phase >= 20:
        enemy_king = board.king(not board.turn)
        
        file = chess.square_file(enemy_king)
        rank = chess.square_rank(enemy_king)
        
        distance_from_center = abs(file - 3.5) + abs(rank - 3.5)
        
        score += distance_from_center * 20
    return score




def evaluate2(board: chess.Board):
    us_idx, them_idx = board_to_halfkp(board.fen())
    us_idx = pad_indices(us_idx)[None, :]
    them_idx = pad_indices(them_idx)[None, :]
    out = float(model({"us_idx":us_idx, "them_idx":them_idx}, training = False)[0,0])
    return out 
   



def game_phase(board:chess.Board):
    
    phase = TOTAL_PHASE

    for piece_type, weight in PHASE_WEIGHTS.items():
        phase -= weight * (
            len(board.pieces(piece_type, chess.WHITE)) +
            len(board.pieces(piece_type, chess.BLACK))
        )

    return phase / TOTAL_PHASE

def board_to_vector(fen):
    board = chess.Board(fen)
    x = np.zeros(768, dtype=np.int8)

    us, them = (chess.WHITE, chess.BLACK) if board.turn == chess.WHITE else (chess.BLACK, chess.WHITE)

    for color_offset, color in ((0, us), (6, them)):
        for piece_type in range(1, 7):
            channel = color_offset + piece_type - 1
            base = channel * 64

            for square in board.pieces(piece_type, color):
                sq = square if board.turn == chess.WHITE else chess.square_mirror(square)
                x[base + sq] = 1

    return x


PIECE_CHAR_TO_TYPE = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4}

def parse_fen_fast(fen):
    placement, turn = fen.split(' ', 2)[:2]
    pieces = []
    king_sq = [None, None]

    rank, file = 7, 0
    for ch in placement:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            is_white = ch.isupper()
            square = rank * 8 + file
            if ch.lower() == 'k':
                king_sq[is_white] = square
            else:
                pieces.append((square, PIECE_CHAR_TO_TYPE[ch.lower()], is_white))
            file += 1

    us_is_white = (turn == 'w')
    return pieces, king_sq, us_is_white

def mirror_sq(square):
    return square ^ 56

def orient(square, perspective_is_white):
    return square if perspective_is_white else mirror_sq(square)

def halfkp_from_parsed(pieces, king_sq, perspective_is_white):
    k = orient(king_sq[perspective_is_white], perspective_is_white)
    indices = []
    for square, ptype, is_white in pieces:
        relative = 0 if is_white == perspective_is_white else 1
        p_idx = relative * 5 + ptype
        sq = orient(square, perspective_is_white)
        indices.append(k * 640 + p_idx * 64 + sq)
    return indices

def board_to_halfkp(fen):
    pieces, king_sq, us_is_white = parse_fen_fast(fen)
    us_idx = halfkp_from_parsed(pieces, king_sq, us_is_white)
    them_idx = halfkp_from_parsed(pieces, king_sq, not us_is_white)
    return np.array(us_idx, dtype=np.int32), np.array(them_idx, dtype=np.int32)

MAX_PIECES = 32

def pad_indices(idx):
    padded = np.full(MAX_PIECES, -1, dtype=np.int32)
    padded[:len(idx)] = idx
    return padded