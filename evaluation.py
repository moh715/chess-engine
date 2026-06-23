import chess
from positions import PIECES_MAP, MG_MAP

def evaluate(board: chess.Board):
    piece_values = {
        chess.PAWN: 100,
        chess.ROOK: 500,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }
    score = 0

    for square, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]

        if piece.color == chess.WHITE:
            score += value
            score += MG_MAP[piece.piece_type][square]
        else:
            score -= value
            score -= MG_MAP[piece.piece_type][chess.square_mirror(square)]

    return score if board.turn == chess.WHITE else -score
