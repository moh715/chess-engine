import chess
import chess.engine
from positions import PIECES_MAP
from typing import Optional

META = 1000000

class Searcher():
    def __init__(self, board: chess.Board, evaluation):
        self.board = board
        self.evaluate = evaluation
        self.tt = {}


    def search(self, max_depth: int):
        """Iterative deepening — returns (best_score, best_move)."""
        assert max_depth >= 1
        best_move = None
        best_score = 0

        for depth in range(1, max_depth + 1):
            best_score, best_move = self._get_best_move(depth)

        return best_score, best_move


    def _get_best_move(self, depth: int):
        """
        One-ply root search: tries every legal move and scores each with
        minmax. Returns (best_score, best_move).
        """
        best_score = float("-inf")
        best_move = None
        alpha = float("-inf")
        beta = float("inf")

        moves = self._sorted_moves(list(self.board.legal_moves))
        for move in moves:
            self.board.push(move)
            score = -self.minmax(-beta, -alpha, depth - 1, ply=1)
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)

        return best_score, best_move


    def minmax(self, alpha: float, beta: float, depth: int, ply: int) -> float:
        """Alpha-beta negamax. Returns a score only."""
        if self.board.is_game_over():
            return self._terminal_score(ply)

        if depth == 0:
            return self.quiesce(alpha, beta, ply)

        key = (self.board._transposition_key(), depth)
        cached = self._tt_lookup(key, alpha, beta)
        if cached is not None:
            return cached

        orig_alpha = alpha
        best_value = float("-inf")

        for move in self._sorted_moves(list(self.board.legal_moves)):
            self.board.push(move)
            value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
            self.board.pop()

            if value > best_value:
                best_value = value
                alpha = max(alpha, value)

            if best_value >= beta:
                break

        self.tt[key] = (best_value, orig_alpha, beta)
        return best_value

    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search. Returns a score only."""
        if self.board.is_game_over():
            return self._terminal_score(ply)

        key = (self.board._transposition_key(), ply)
        cached = self._tt_lookup(key, alpha, beta)
        if cached is not None:
            return cached

        orig_alpha = alpha

        if self.board.is_check():
            best_value = float("-inf")
            moves = list(self.board.legal_moves)
        else:
            stand_pat = self.evaluate(self.board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best_value = stand_pat
            moves = [m for m in self.board.legal_moves
                     if self.board.is_capture(m) or m.promotion]

        for move in self._sorted_moves(moves):
            self.board.push(move)
            value = -self.quiesce(-beta, -alpha, ply + 1)
            self.board.pop()

            if value > best_value:
                best_value = value
                alpha = max(alpha, best_value)

            if alpha >= beta:
                break

        self.tt[key] = (best_value, orig_alpha, beta)
        return best_value


    def _tt_lookup(self, key, alpha: float, beta: float):
        """
        Returns a usable cached score or None.
        Entries are stored as (value, lo, hi) where lo/hi are the
        alpha/beta bounds the value was searched under.
        """
        if key not in self.tt:
            return None
        value, lo, hi = self.tt[key]
        if lo <= value <= hi and lo >= alpha and hi <= beta:
            return value
        if value >= beta:
            return value
        if value <= alpha:
            return value
        return None


    def _sorted_moves(self, moves):
        return sorted(moves, key=self._move_tactical_score, reverse=True)

    def _capture_score(self, move) -> int:
        victim = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        return 100 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def _move_tactical_score(self, move) -> int:
        if self.board.is_capture(move):
            return self._capture_score(move)
        if move.promotion:
            return 1000 + (move.promotion == chess.QUEEN) * 900
        if self.board.gives_check(move):
            return 800
        return 0


    def _terminal_score(self, ply: int) -> float:
        if self.board.is_variant_loss():
            return -META + ply
        if self.board.is_variant_win():
            return META - ply
        return 0            
        
def evaluate(board:chess.Board):
    if board.is_game_over():
        outcome = board.outcome()
        
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return META
            elif outcome.winner == chess.BLACK:
                return -META
            else:
                return 0
    piece_values = {
    chess.PAWN: 100,
    chess.ROOK: 500,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.QUEEN: 900,
    chess.KING: 20000
    }
    score = 0
    
    for square, piece in board.piece_map().items():
        value = piece_values[piece.piece_type]

        if piece.color == chess.WHITE:
            score += value
            score += PIECES_MAP[piece.piece_type][square]
        else:
            score -= value
            score -= PIECES_MAP[piece.piece_type][chess.square_mirror(square)]

    return score if board.turn == chess.WHITE else -score


if __name__ == "__main__":
    board = chess.Board("1Qbqkbr1/1ppppppp/8/5P2/8/5N2/PPP2PPP/RNB1KB1R b KQ - 0 7")
    print(Searcher(board, evaluate).minmax())