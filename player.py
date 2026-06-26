from collections import defaultdict
from time import perf_counter
from evaluation import evaluate
import chess

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
        self.beta_cutof = 0
        self.history = defaultdict(int)
        self.WINDOW_MARGIN = 50
        self.aspr_fail = 0
        self.tactical_scores = defaultdict(int)
        self.time_sort    = 0.0   
        self.time_eval    = 0.0   
        self.time_quiesce = 0.0   
        self.time_tt      = 0.0   


    def print_profile(self):
        """Print a tidy summary of where time was spent."""
        total = self.time_sort + self.time_eval + self.time_quiesce + self.time_tt
        rows = [
            ("move sorting",  self.time_sort),
            ("evaluation",    self.time_eval),
            ("quiescence",    self.time_quiesce),
            ("TT lookup/store", self.time_tt),
        ]
        print(f"\n{'Section':<22} {'Seconds':>9}  {'Share':>7}")
        print("-" * 42)
        for label, t in rows:
            pct = (t / total * 100) if total else 0
            print(f"  {label:<20} {t:>9.4f}  {pct:>6.1f}%")
        print(f"  {'TOTAL':<20} {total:>9.4f}")
        print(f"\n  nodes={self.nodes:,}  \n tt_hits={self.tt_hits:,} \n "
              f"tt_lookups={self.tt_lookups:,}  \nbeta_cuts={self.beta_cutof:,}")


    def search(self, max_depth: int):
        """Iterative deepening — returns (best_score, best_move)."""
        assert max_depth >= 1
        best_move = None
        best_score = 0
        alpha = -META
        beta  = META

        for depth in range(1, max_depth + 1):
            if depth == 1:
                best_score, best_move = self._get_best_move(depth, alpha, beta)
            else:
                margin = self.WINDOW_MARGIN
                while True:
                    alpha = best_score - margin
                    beta  = best_score + margin

                    score, move = self._get_best_move(depth, alpha, beta)

                    if score <= alpha or score >= beta:
                        self.aspr_fail += 1
                        margin *= 2
                    else:
                        best_score = score
                        best_move  = move
                        break

        return best_score, best_move

    def _get_best_move(self, depth: int, alpha: int, beta: int):
        """Root search: scores every legal move with negamax."""
        best_score = float("-inf")
        best_move  = None

        t = perf_counter()
        moves = self._sorted_moves(list(self.board.legal_moves), depth)
        self.time_sort += perf_counter() - t

        for move in moves:
            self.board.push(move)
            score = -self.minmax(-beta, -alpha, depth - 1, ply=1)
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move  = move
                alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score, best_move


    def minmax(self, alpha: float, beta: float, depth: int, ply: int,
               allow_null: bool = True) -> float:
        """Alpha-beta negamax. Returns a score only."""

        if self.board.is_game_over():
            return self._terminal_score(ply)
        if self.board.is_repetition():
            return -100
        if depth == 0:
            t = perf_counter()
            v = self.quiesce(alpha, beta, ply)
            self.time_quiesce += perf_counter() - t
            return v

        key = self.board._transposition_key()

        t = perf_counter()
        cached = self._tt_lookup(key, depth, alpha, beta, True)
        self.time_tt += perf_counter() - t

        if cached is not None:
            return cached

        self.nodes += 1
        orig_alpha = alpha

        if (allow_null
                and depth >= 3
                and not self.board.is_check()
                and self._has_pieces_left()):
            R = 2 if depth <= 6 else 3
            self.board.push(chess.Move.null())
            score = -self.minmax(-beta, -beta + 1,
                                 depth - 1 - R, ply + 1, False)
            self.board.pop()
            if score >= beta:
                return beta

        best_value = float("-inf")
        best_move  = None
        best = self._get_best_cached()
        moves = [(m, self._move_tactical_score(m, best, ply)) for m in self.board.legal_moves] 

        number = 0
        while moves:
            max_move = moves[0]
            for m in moves:
                if m[1] > max_move[1]:
                    max_move = m
            move = max_move[0]            
            self.board.push(move)
            if number == 0:
                value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
            else:
                reduction = self._get_reduction(move, number, depth)
                value = -self.minmax(-alpha - 1, -alpha,
                                     depth - 1 - reduction, ply + 1)
                if alpha < value < beta:
                    value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
            self.board.pop()

            if value > best_value:
                best_value = value
                best_move  = move
                alpha = max(alpha, value)

            if best_value >= beta:
                self.beta_cutof += 1
                if not self.board.is_capture(move):
                    if move != self.killer[ply][0]:
                        self.killer[ply][1] = self.killer[ply][0]
                        self.killer[ply][0] = move
                    self.history[move] += depth * depth
                break
            number += 1
            idx = moves.index(max_move)
            moves[idx] = moves[-1]
            moves.pop()
        t = perf_counter()
        self._cache(key, depth, best_value, best_move, orig_alpha, beta, True)
        self.time_tt += perf_counter() - t

        return best_value


    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search. Returns a score only."""
        self.nodes += 1
        margin = 900

        if self.board.is_game_over():
            return self._terminal_score(ply)

        key = self.board._transposition_key()

        t = perf_counter()
        cached = self._tt_lookup(key, 0, alpha, beta, False)
        self.time_tt += perf_counter() - t

        if cached is not None:
            return cached

        orig_alpha = alpha
        best_move  = None   
        best = self._get_best_cached()
        if self.board.is_check():
            best_value = float("-inf")
            moves = [(m, self._move_tactical_score(m, best, ply)) for m in self.board.legal_moves]
        else:
            t = perf_counter()
            stand_pat = self.evaluate(self.board)
            self.time_eval += perf_counter() - t

            if stand_pat >= beta:
                return stand_pat
            alpha      = max(alpha, stand_pat)
            best_value = stand_pat
            
            if stand_pat + margin < alpha:
                return stand_pat
            moves      = [(m, self._move_tactical_score(m, best, ply)) for m in self.board.legal_moves
                          if self.board.is_capture(m) or m.promotion]

        

        first_move = True
        while moves:
            max_move = moves[0]
            for m in moves:
                if m[1] > max_move[1]:
                    max_move = m
            move = max_move[0]
            self.board.push(move)
            if first_move:
                value      = -self.quiesce(-beta, -alpha, ply + 1)
                first_move = False
            else:
                value = -self.quiesce(-alpha - 1, -alpha, ply + 1)
                if alpha < value < beta:
                    value = -self.quiesce(-beta, -alpha, ply + 1)
            self.board.pop()

            if value > best_value:
                best_value = value
                best_move  = move
                alpha      = max(alpha, best_value)

            if alpha >= beta:
                self.beta_cutof += 1
                break
            idx = moves.index(max_move)
            moves[idx] = moves[-1]
            moves.pop()
        t = perf_counter()
        self._cache(key, 0, best_value, best_move, orig_alpha, beta, False)
        self.time_tt += perf_counter() - t

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
        if flag == LOWERBOUND and value >= beta:
            return value
        if flag == UPPERBOUND and value <= alpha:
            return value
        return None

    def _cache(self, key, depth, best_value, best_move, alpha, beta, is_minmax):
        table = self.tt if is_minmax else self.qtt

        if best_value <= alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT

        old = table.get(key)
        if old is None or depth >= old[2]:
            table[key] = (best_value, best_move, depth, flag)

    def _get_best_cached(self):
        key   = self.board._transposition_key()
        entry = self.tt.get(key)
        best  = entry[1] if entry else None
        return best

        
    def _sorted_moves(self, moves, ply):
        key   = self.board._transposition_key()
        entry = self.tt.get(key)
        best  = entry[1] if entry else None
        
        return sorted(
            moves,
            key=lambda m: self._move_tactical_score(m, best, ply),
            reverse=True,
        )

    def _see(self, move: chess.Move) -> int:
            piece_values = {
                chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
            }
            
            victim = self.board.piece_at(move.to_square)
            if not victim:
                return 0
                
            gain = piece_values[victim.piece_type]
            
            self.board.push(move)
            
           
            next_mover_color = self.board.turn
            all_attackers = self.board.attackers(next_mover_color, move.to_square)
        
            best_attacker_sq = None
            lowest_value     = float("inf")
            for sq in all_attackers:
                p = self.board.piece_at(sq)
                if chess.Move(sq, move.to_square) not in self.board.legal_moves:
                    continue
                if p and piece_values[p.piece_type] < lowest_value:
                    lowest_value     = piece_values[p.piece_type]
                    best_attacker_sq = sq
                    
            if best_attacker_sq is not None:
                recapture = chess.Move(best_attacker_sq, move.to_square)
                # Opponent recaptures only if profitable; max(0, ...) models
                # their option to simply not recapture a losing exchange.
                score = gain - max(0, self._see(recapture))   # ← was max(-100, gain - ...)
            else:
                score = gain   # nothing can recapture — we keep everything we captured
                
            self.board.pop()
            return score


    def _mvv_lva(self, move):
        victim   = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100,
        }
        return 100 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]


    
    def _move_tactical_score(self, move: chess.Move, best_move, ply: int) -> int:
        score  = 0
        killer = self.killer[ply] if ply < MAX_DEPTH else []
        key = (self.board._transposition_key(), move)
        if move == best_move:
            return 10000
        if move in killer:
            score += 700
        score += self.history[move]
        static = self.tactical_scores.get(key)
        if static:
            score += static
            return score
        static = 0
        if self.board.is_castling(move):
            static += 90
        if move.promotion:
            static += 100 + (move.promotion == chess.QUEEN) * 90
        if self.board.is_capture(move):
            static += self._mvv_lva(move)
            # if score < 500:
            #     static += self._see(move)
        self.tactical_scores[key] = static
        return score + static


    def _get_reduction(self, move: chess.Move, number: int, depth: int) -> int:
        if self.board.is_capture(move):   return 0
        if self.board.gives_check(move):  return 0
        if number <= 5:                   return 0
        if depth  <= 3:                   return 0
        if number > 20 and depth > 6:     return 4
        if number > 10 and depth > 5:     return 3
        return 1

    def _terminal_score(self, ply: int) -> float:
        if self.board.is_checkmate():
            return -META + ply
        return 0

    def _has_pieces_left(self) -> bool:
        """True when the side to move has at least one piece beyond king/pawns."""
        us = self.board.turn
        non_pawn_pieces = (
            self.board.pieces(chess.KNIGHT, us) |
            self.board.pieces(chess.BISHOP, us) |
            self.board.pieces(chess.ROOK,   us) |
            self.board.pieces(chess.QUEEN,  us)
        )
        return bool(non_pawn_pieces)

if __name__ == "__main__":
    board = chess.Board()
    searcher = Searcher(board, evaluate)
    sum = 0
    while not board.is_game_over():
        _, move = searcher.search(6)
        board.push(move)
        sum += searcher.nodes
        searcher.nodes = 0 
    print(sum)