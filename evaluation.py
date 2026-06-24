import chess
from positions import EG_MAP, MG_MAP

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
    if board.is_repetition():
            score += -200
    if phase >= 20:
        enemy_king = board.king(not board.turn)
        
        file = chess.square_file(enemy_king)
        rank = chess.square_rank(enemy_king)
        
        distance_from_center = abs(file - 3.5) + abs(rank - 3.5)
        
        score += distance_from_center * 20
    return score



def evaluate2(board: chess.Board):
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
    if board.is_repetition():
            score += -200
    return score



def game_phase(board):
    
    phase = TOTAL_PHASE

    for piece_type, weight in PHASE_WEIGHTS.items():
        phase -= weight * (
            len(board.pieces(piece_type, chess.WHITE)) +
            len(board.pieces(piece_type, chess.BLACK))
        )

    return phase / TOTAL_PHASE