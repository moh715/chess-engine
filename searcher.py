from collections import defaultdict
from time import perf_counter
import heapq
from evaluation import Evaluation, Handcrafted
import bulletchess as bc
from evaluation import Handcrafted, NNEvaluation, Evaluation
from seee import SEEEvaluator

META = 1e7
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2
MAX_DEPTH = 20


class Searcher():
    def __init__(self, board: bc.Board, evaluation: Evaluation):
        self.board = board
        self.evaluate = evaluation
        self.evaluate.set_board(board)        
        self.seee = SEEEvaluator(board)
        self.see = self.seee.see
        self.tt = {}
        self.qtt = {}
        self.killer = [[None, None] for _ in range(MAX_DEPTH)]
        self.history = defaultdict(int)
        self.WINDOW_MARGIN = self.evaluate.window_margin
        self.pieces_values ={
                    bc.PAWN: 100, bc.KNIGHT: 320, bc.BISHOP: 300,
                    bc.ROOK: 500, bc.QUEEN: 900, bc.KING: 100,
                }
        self.nodes = 0
        self.tt_hits = 0
        self.tt_lookups = 0
        self.beta_cutof = 0
        self.aspr_fail = 0
        self.time_sort    = 0.0
        self.time_eval    = 0.0
        self.time_quiesce = 0.0
        self.time_tt      = 0.0
        

    def print_profile(self):
        """Print a tidy summary of where time was spent."""
        print("nodes: ", self.nodes)
        print("beta cutofs: ", self.beta_cutof)
        print("tt_lookups: ", self.tt_lookups)
        print("tt_hits: ", self.tt_hits)
        print("TT hit rate:", round(100 * (self.tt_hits / self.tt_lookups), 2), "%")
        print("lookups/cache time", self.time_tt)
        print("aspr_fails: ", self.aspr_fail)


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
                    # Clamp alpha/beta to META so we don't search outside the valid score range
                    alpha = max(-META, best_score - margin)
                    beta  = min(META, best_score + margin)

                    score, move = self._get_best_move(depth, alpha, beta)

                    if score <= alpha or score >= beta:
                        self.aspr_fail += 1
                        margin *= 2
                        
                        
                        if alpha <= -META and beta >= META:
                            best_score = score
                            best_move  = move
                            break
                    else:
                        best_score = score
                        best_move  = move
                        break
    
        return best_score, best_move

    def _get_best_move(self, depth: int, alpha: int, beta: int) -> tuple[float, bc.Move]:
        """Root search: scores every legal move with negamax."""
        best_score = float("-inf")
        best_move  = None
        moves = self._sorted_moves(list(self.board.legal_moves()), depth)
        for move in moves:
            self.evaluate.do(move)
            self.board.apply(move)
            score = -self.minmax(-beta, -alpha, depth - 1, ply=1)
            self.board.undo()
            self.evaluate.undo()
            if score > best_score:
                best_score = score
                best_move  = move
                alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score, best_move


    def minmax(self, alpha: float, beta: float, depth: int, ply: int,
               allow_null: bool = True) -> float:

        if self.board in bc.MATE:
            return self._terminal_score(ply)
        if self.board in bc.THREEFOLD_REPETITION:
            return -1
        if depth == 0:
            v = self.quiesce(alpha, beta, ply)
            return v

        key = hash(self.board)
        cached = self._tt_lookup(key, depth, alpha, beta, True)
        

        if cached is not None:
            return cached

        self.nodes += 1
        orig_alpha = alpha

        if (allow_null
            and depth >= 3
            and not self.board in bc.CHECK
            and self._has_pieces_left()):
            R = 2 if depth <= 6 else 3
            self.board.apply(None)
            score = -self.minmax(-beta, -beta + 1,
                                 depth - 1 - R, ply + 1, False)
            self.board.undo()
            if score >= beta:
                return beta

        best_value = float("-inf")
        best_move  = None
        entry = self.tt.get(key)
        best = entry[1] if entry else None
        moves = [ (-self._move_tactical_score(m, best, ply), i,m) for i, m in enumerate(self.board.legal_moves())]
        heapq.heapify(moves)
        number = 0

        while moves:
            _,_, move = moves[0]
            reduction = self._get_reduction(move, number, depth)
            self.evaluate.do(move)
            self.board.apply(move)
            if number == 0:
                value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)
            else:
                value = -self.minmax(-alpha - 1, -alpha,
                                     depth - 1 - reduction, ply + 1)
                if alpha < value < beta:
                    value = -self.minmax(-beta, -alpha, depth - 1, ply + 1)

            self.board.undo()
            self.evaluate.undo()
            number += 1
            if value > best_value:
                best_value = value
                best_move  = move
                alpha = max(alpha, value)

            if best_value >= beta:
                self.beta_cutof += 1
                if not move.is_capture(self.board):
                    if move != self.killer[ply][0]:
                        self.killer[ply][1] = self.killer[ply][0]
                        self.killer[ply][0] = move
                    self.history[move] += depth * depth
                break
            heapq.heappop(moves)
        self._cache(key, depth, best_value, best_move, orig_alpha, beta, True)
        

        return best_value


    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if self.board in bc.MATE:
            return self._terminal_score(ply)

        key = hash(self.board)
        cached = self._tt_lookup(key, 0, alpha, beta, False)
        
        if cached is not None:
            return cached

        orig_alpha = alpha
        pv = self.pieces_values

        if self.board in bc.CHECK:
            best_value = float("-inf")
            moves = self.board.legal_moves()
        else:
            stand_pat = self.evaluate()

            if stand_pat >= beta:
                return stand_pat

            alpha = max(alpha, stand_pat)
            best_value = stand_pat

            if stand_pat + self.evaluate.queen < alpha:
                self._cache(key, 0, best_value, None, orig_alpha, beta, False)
                return best_value

            moves = []
            for move in self.board.legal_moves():
                if not (move.is_capture(self.board) or move.promotion):
                    continue

                victim = self.board[move.destination]
                gain = pv[victim.piece_type] if victim else 0
                if move.promotion:
                    gain += pv[move.promotion]

                if stand_pat + gain + 200 < alpha:
                    continue
                if self.see(move) < 0:
                    continue
                moves.append(move)

        entry = self.tt.get(key)
        best  = entry[1] if entry else None
        moves = [(-self._move_tactical_score(m, best, ply), i, m)
                for i, m in enumerate(moves)]
        heapq.heapify(moves)

        best_move  = None
        first_move = True
        while moves:
            _, _, move = moves[0]

            self.evaluate.do(move)
            self.board.apply(move)
            if first_move:
                value      = -self.quiesce(-beta, -alpha, ply + 1)
                first_move = False
            else:
                value = -self.quiesce(-alpha - 1, -alpha, ply + 1)
                if alpha < value < beta:
                    value = -self.quiesce(-beta, -alpha, ply + 1)
            self.board.undo()
            self.evaluate.undo()

            if value > best_value:
                best_value = value
                best_move  = move
                alpha      = max(alpha, best_value)

            if alpha >= beta:
                self.beta_cutof += 1
                break
            heapq.heappop(moves)
        self._cache(key, 0, best_value, best_move, orig_alpha, beta, False)
        
        return best_value

    def reset(self, board: bc.Board):
        """Rebind this searcher to a new game/board, clearing all per-game state."""
        self.board = board
        self.evaluate.set_board(board)
        self.seee = SEEEvaluator(board)
        self.see = self.seee.see
    
        self.tt.clear()
        self.qtt.clear()
        self.killer = [[None, None] for _ in range(MAX_DEPTH)]
        self.history = defaultdict(int)
    
        self.nodes = 0
        self.tt_hits = 0
        self.tt_lookups = 0
        self.beta_cutof = 0
        self.aspr_fail = 0

        
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


    def _sorted_moves(self, moves, ply):
        key   = hash(self.board)
        entry = self.tt.get(key)
        best  = entry[1] if entry else None

        return sorted(
            moves,
            key=lambda m: self._move_tactical_score(m, best, ply),
            reverse=True,
        )

    def _capture_score(self, move:bc.Move) -> int:
        victim   = self.board[move.destination]
        attacker = self.board[move.origin]
        if victim is None or attacker is None:
            return 0
        piece_values = {
            bc.PAWN: 100, bc.KNIGHT: 320, bc.BISHOP: 330,
            bc.ROOK: 500, bc.QUEEN: 900, bc.KING: 0,
        }
        return piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def _move_tactical_score(self, move: bc.Move, best_move, ply: int) -> int:
        score  = 0
        killer = self.killer[ply] if ply < MAX_DEPTH else []

        if move == best_move:
            score += 1000
        if move in killer:
            score += 700
        
        score += self.see(move)
        # if move.is_castling(self.board):
        #     score += 50
        if move.promotion:
            score += 100 + (move.promotion == bc.QUEEN) * 90
        score += self.history[move]
        return score


    def _get_reduction(self, move: bc.Move, number: int, depth: int) -> int:
        if move.is_capture(self.board):
            return 0
        if self._gives_check(move):
            return 0
        if number <= 5:
            return 0
        if depth  <= 3:
            return 0
        if number > 25:
            return depth - 1
        if number > 10:
            return depth - 2
        return 2

    def _terminal_score(self, ply: int) -> float:
        if self.board in bc.CHECKMATE:
            return -META + ply
        return 0


    def _gives_check(self, move: bc.Move) -> bool:
            self.board.apply(move)
            result = self.board in bc.CHECK
            self.board.undo()
            return result

    
    def _has_pieces_left(self) -> bool:
        """True when the side to move has at least one piece beyond king/pawns."""
        us = self.board.turn
        non_pawn_pieces = (
            self.board[us, bc.KNIGHT] |
            self.board[us, bc.BISHOP] |
            self.board[us, bc.ROOK] |
            self.board[us, bc.QUEEN]
        )
        return bool(non_pawn_pieces)


if __name__ == "__main__":
    import cProfile
    import pstats
    
    board = bc.Board()
    def benchmark():
        searcher = Searcher(board, NNEvaluation())
        print(searcher.search(7))

    
    profiler = cProfile.Profile()
    profiler.enable()
    
    benchmark()
    
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(25)
