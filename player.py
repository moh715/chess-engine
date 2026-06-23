import chess
import chess.engine
from positions import PIECES_MAP
from typing import Optional
from collections import defaultdict
from evaluation import evaluate

META = 1000000

class Searcher():
    def __init__(self, board: chess.Board, evaluation):
        self.board = board
        self.evaluate = evaluation
        self.tt = defaultdict(dict)
        self.nodes = 0
        self.tt_hits = 0
        self.tt_lookups = 0 

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
        if self.board.is_repetition():
            return -10
        if depth == 0:
            return self.quiesce(alpha, beta, ply)

        key = (self.board._transposition_key(), depth)
        cached = self._tt_lookup(key, alpha, beta)
        if cached is not None:
            return cached
        self.nodes += 1
        orig_alpha = alpha
        best_value = float("-inf")
        best_move = None
        for move in self._sorted_moves(list(self.board.legal_moves)):
            self.board.push(move)
            value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
            self.board.pop()
            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, value)

            if best_value >= beta:
                break

        self.tt[key[0]][key[1]] = (best_value,best_move, orig_alpha, beta)
        return best_value

    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search. Returns a score only."""
        if self.board.is_game_over():
            return self._terminal_score(ply)
        if self.board.is_repetition():
            return -10
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
            best_move = None
            moves = [m for m in self.board.legal_moves
                     if self.board.is_capture(m) or m.promotion]

        for move in self._sorted_moves(moves):
            self.board.push(move)
            value = -self.quiesce(-beta, -alpha, ply + 1)
            self.board.pop()
            self.nodes += 1
            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, best_value)

            if alpha >= beta:
                break

        self.tt[key[0]][key[1]] = (best_value,best_move , orig_alpha, beta)
        return best_value


    def _tt_lookup(self, key, alpha: float, beta: float):
        """
        Returns a usable cached score or None.
        Entries are stored as (value, lo, hi) where lo/hi are the
        alpha/beta bounds the value was searched under.
        """
        self.tt_lookups += 1
        if key[0] not in self.tt:
            return None
        depth = max(self.tt[key[0]].keys())
        if key[1] > depth:
            return None
        value,_, lo, hi = self.tt[key[0]][depth]
        if lo <= value <= hi and lo >= alpha and hi <= beta:
            self.tt_hits += 1
            return value
        
        if value >= beta:
            self.tt_hits += 1
            return value
        
        if value <= alpha:
            self.tt_hits += 1
            return value
        
        return None


    def _sorted_moves(self, moves):
        best_move = None
        if self.board._transposition_key() in self.tt:
             depth = max(self.tt[self.board._transposition_key()].keys())
             best_move = self.tt[self.board._transposition_key()][depth][1]
        return sorted(moves, key = lambda m: (
                    m == best_move,
                    self._move_tactical_score(m)
                ), reverse=True)
        # return sorted(moves, key=self._move_tactical_score, reverse=True)

    def _capture_score(self, move) -> int:
        victim = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        return 100 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def _move_tactical_score(self, move) -> int:
        # self.board.push(move)
        # score = self._tt_lookup((self.board._transposition_key(), 2), -META, META)
        # self.board.pop()
        # if score:
        #     return score
        score = 0
        if self.board.is_capture(move):
            score += self._capture_score(move)
        if move.promotion:
            score += 100 + (move.promotion == chess.QUEEN) * 90
        if self.board.gives_check(move):
            score +=  80
        return score


    def _terminal_score(self, ply: int) -> float:
        if self.board.is_variant_loss():
            return -META + ply
        if self.board.is_variant_win():
            return META - ply
        if self.board.can_claim_threefold_repetition():
           return -1 
        return 0            
        



if __name__ == "__main__":
    # board = chess.Board()
    # searcher = Searcher(board)
    board = chess.Board()
    searcher = Searcher(board, evaluate)
    sum = 0
    while not board.is_game_over():
        _, move = searcher.search(6)
        board.push(move)
        sum += searcher.nodes
        searcher.nodes = 0 
    print(sum)