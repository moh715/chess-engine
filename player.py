import chess
import threading
from multiprocessing import Pool



class Seracher():
    def __init__(self, board, evaluation):
        self.board = board
        self.evaluate = evaluation
        self.moves = []


    def minmax(self, alpha=float("-inf"), beta=float("inf"), depth=3):
        if depth == 0:
            return self.evaluate(self.board), []
        if self.board.is_checkmate():
            return float("-inf"), []
        best_value = float("-inf")
        best_moves = []
        moves = list(self.board.legal_moves)
        moves.sort(key=lambda m: self._move_tactical_score(m), reverse=True)
        for move in moves:
            self.board.push(move)
            value, pv = self.minmax(-beta, -alpha, depth-1)
            value *= -1
            self.board.pop()
            if value > best_value:
                best_value = value
                best_moves = [move] + pv
                alpha = max(alpha, value)
            if best_value >= beta:
                break 
        return alpha, best_moves
    

    def _move_tactical_score(self, move):
        if self.board.is_capture(move):
            return self._capture_score(move)
        elif move.promotion:
            return 1000 + (move.promotion == chess.QUEEN) * 900
        elif self.board.gives_check(move):
            return 800
        return 0
    
    def _capture_score(self, move):
        victim = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        return piece_values[victim.piece_type] - piece_values[attacker.piece_type] / 100



            
def evaluate(board:chess.Board):
    piece_values = {
    chess.PAWN: 100,
    chess.ROOK: 500,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.QUEEN: 900,
    chess.KING: 20000
    }
    score = 0
    for piece_type in chess.PIECE_TYPES:
            pieces_mask = board.pieces_mask(piece_type, chess.WHITE)
            score += chess.popcount(pieces_mask) * piece_values[piece_type]
            pieces_mask = board.pieces_mask(piece_type, chess.BLACK)
            score -= chess.popcount(pieces_mask) * piece_values[piece_type]
    return score

board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
seracher = Seracher(board, evaluate)
for move in seracher.minmax(depth=6)[1]:
    board.push(move)  
    print("\n\n")
    print(board)
