import chess
import chess.engine
from positions import PIECES_MAP
from typing import Optional
from collections import defaultdict
from evaluation import evaluate

META = 1e7
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2
MAX_DEPTH = 20


class Searcher():
    def __init__(self, board: chess.Board, evaluation):
        self.board = board
        self.evaluate = evaluation
        self.tt = {}
        self.qtt = {}
        self.killer = [[None, None] for _ in range(MAX_DEPTH)]
        self.nodes = 0
        self.tt_hits = 0
        self.tt_lookups = 0 
        self.history = defaultdict(int)

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

        moves = self._sorted_moves(list(self.board.legal_moves), depth)
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
            return -100
        if depth == 0:
            return self.quiesce(alpha, beta, ply)

        key = self.board._transposition_key()
        cached = self._tt_lookup(key, depth,alpha, beta, True)
        if cached is not None:
            return cached
        self.nodes += 1
        orig_alpha = alpha
        best_value = float("-inf")
        best_move = None
        first_move = True
        for move in self._sorted_moves(list(self.board.legal_moves), ply):
            self.board.push(move)
            
            if first_move:
                value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
                first_move = False
            else:
                value = -self.minmax(-alpha - 1, -alpha, depth - 1, ply + 1)
        
                if alpha < value < beta:
                    value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
        
            self.board.pop()
            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, value)

            if best_value >= beta:
                if move != self.killer[ply][0] and not self.board.is_capture(move):
                        self.killer[ply][1] = self.killer[depth][0]
                        self.killer[ply][0] = move
                        self.history[move] += depth* depth
                break
        self._cache(
            key,
            depth,
            best_value,
            best_move,
            orig_alpha,
            beta,
            True
        )
        return best_value

    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search. Returns a score only."""
        self.nodes += 1
        if self.board.is_game_over():
            return self._terminal_score(ply)
        key = self.board._transposition_key()
        cached = self._tt_lookup(key, 0, alpha, beta, False)
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

        for move in self._sorted_moves(moves, ply):
            self.board.push(move)
            value = -self.quiesce(-beta, -alpha, ply + 1)
            self.board.pop()
            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, best_value)

            if alpha >= beta:
                break
        self._cache(
            key,
            0,
            best_value,
            best_move,
            orig_alpha,
            beta,
            False
        )
        return best_value


    def _tt_lookup(self, key, depth, alpha, beta, is_minmax):
        self.tt_lookups += 1
        table = self.tt if is_minmax else self.qtt
        entry = table.get(key)
        if entry is None:
            return None
    
        value, best_move, stored_depth, flag = entry
    
        if stored_depth < depth:
            return None
    
        self.tt_hits += 1
    
        if flag == EXACT:
            return value
    
        if flag == LOWERBOUND:
            if value >= beta:
                return value
    
        elif flag == UPPERBOUND:
            if value <= alpha:
                return value
    
        return None

    def _cache(self, key, depth, best_value, best_move, alpha, beta, is_minmax):
        flag = EXACT
        table = self.tt if is_minmax else self.qtt 
        if best_value <= alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
    
        old = table.get(key)
    
        if old is None or depth >= old[2]:
            table[key] = (
                best_value,
                best_move,
                depth,
                flag
            )


            
    def _sorted_moves(self, moves, ply):
        return sorted(
            moves,
            key=lambda m:self._move_tactical_score(m, ply),
            reverse=True
        )    
    def _capture_score(self, move) -> int:
        victim = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        return 100 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def _move_tactical_score(self, move, ply) -> int:
        score = 0
        best_move = None
        entry = self.tt.get(self.board._transposition_key())
        killer  = self.killer[ply] if ply < MAX_DEPTH else []
        if entry is not None:
            best_move = entry[1]
        if move == best_move:
            score += 1000
        if move in killer:
           score += 700 
        if self.board.is_capture(move):
            score += self._capture_score(move)
        if self.board.is_castling(move):
            score += 50
        if move.promotion:
            score += 100 + (move.promotion == chess.QUEEN) * 90
        if self.board.gives_check(move):
            score +=  80
        score += self.history[move]
        return score


    def _terminal_score(self, ply: int) -> float:
        if self.board.is_checkmate():
                return -META + ply
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