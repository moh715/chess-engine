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
    x = np.expand_dims(board_to_vector(board.fen()), axis=0)
    out =  float(model(x, training=False)[0, 0])
    return out + evaluate(board)
   



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

