from collections import defaultdict
from time import perf_counter
import chess
import cython_chess
import heapq
from evaluation import Handcrafted

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
        self.history = defaultdict(int)
        self.WINDOW_MARGIN = evaluation.window_margin
        self.pieces_values ={
                    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 300,
                    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 100,
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
        print("sort time: ", self.time_sort)
        print("evaluation time: ", self.time_eval)
        print("quiesce time: ", self.time_quiesce)


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
        moves = self._sorted_moves(list(cython_chess.generate_legal_moves(self.board,chess.BB_ALL,chess.BB_ALL)), depth)
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
            return -1
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
        entry = self.tt.get(key)
        best = entry[1] if entry else None
        moves = [ (-self._move_tactical_score(m, best, ply), i,m) for i, m in enumerate(cython_chess.generate_legal_moves(self.board,chess.BB_ALL,chess.BB_ALL))]
        heapq.heapify(moves)
        number = 0

        while moves:
            _,_, move = moves[0]
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
            number += 1
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
            heapq.heappop(moves)

        t = perf_counter()
        self._cache(key, depth, best_value, best_move, orig_alpha, beta, True)
        self.time_tt += perf_counter() - t

        return best_value


    def quiesce(self, alpha: float, beta: float, ply: int) -> float:
        """Quiescence search. Returns a score only."""
        self.nodes += 1
        if self.board.is_game_over():
            return self._terminal_score(ply)

        key = self.board._transposition_key()

        t = perf_counter()
        cached = self._tt_lookup(key, 0, alpha, beta, False)
        self.time_tt += perf_counter() - t
        if cached is not None:
            return cached

        orig_alpha = alpha
        pv = self.pieces_values

        if self.board.is_check():
            best_value = float("-inf")
            moves = list(cython_chess.generate_legal_moves(
                self.board, chess.BB_ALL, chess.BB_ALL))
        else:
            t = perf_counter()
            stand_pat = self.evaluate(self.board)
            self.time_eval += perf_counter() - t

            if stand_pat >= beta:
                return stand_pat

            alpha = max(alpha, stand_pat)
            best_value = stand_pat

            if stand_pat + self.evaluate.queen< alpha:
                self._cache(key, 0, best_value, None, orig_alpha, beta, False)
                return best_value

            moves = []
            for move in cython_chess.generate_legal_moves(self.board, chess.BB_ALL, chess.BB_ALL):
                if not (self.board.is_capture(move) or move.promotion):
                    continue

                victim = self.board.piece_at(move.to_square)
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
            heapq.heappop(moves)

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


    def _sorted_moves(self, moves, ply):
        key   = self.board._transposition_key()
        entry = self.tt.get(key)
        best  = entry[1] if entry else None

        return sorted(
            moves,
            key=lambda m: self._move_tactical_score(m, best, ply),
            reverse=True,
        )

    def _capture_score(self, move) -> int:
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

        if move == best_move:
            score += 1000
        if move in killer:
            score += 700
        score += self.see(move)
        if self.board.is_castling(move):
            score += 50
        if move.promotion:
            score += 100 + (move.promotion == chess.QUEEN) * 90
        # if self.board.gives_check(move):
        #     score += 80
        score += self.history[move]
        return score


    def _get_reduction(self, move: chess.Move, number: int, depth: int) -> int:
        if self.board.is_capture(move):
            return 0
        if self.board.gives_check(move):
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


    def _attacks_to_sq(self, piece_type: int, color: bool,
                       from_sq: int, to_sq: int, occ: int) -> bool:
        """
        Does a piece of (piece_type, color) at from_sq attack to_sq
        given the occupancy bitboard occ?

        Uses python-chess's precomputed attack tables so that:
          - sliders respect the current occ  (x-ray awareness)
          - no board.push / board.pop needed
        """
        if piece_type == chess.PAWN:
            return bool(chess.BB_PAWN_ATTACKS[color][from_sq] & chess.BB_SQUARES[to_sq])

        if piece_type == chess.KNIGHT:
            return bool(chess.BB_KNIGHT_ATTACKS[from_sq] & chess.BB_SQUARES[to_sq])

        if piece_type == chess.KING:
            return bool(chess.BB_KING_ATTACKS[from_sq] & chess.BB_SQUARES[to_sq])

        to_bb = chess.BB_SQUARES[to_sq]

        if piece_type == chess.BISHOP:
            return bool(
                chess.BB_DIAG_ATTACKS[from_sq][occ & chess.BB_DIAG_MASKS[from_sq]] & to_bb
            )

        if piece_type == chess.ROOK:
            return bool(
                (chess.BB_RANK_ATTACKS[from_sq][occ & chess.BB_RANK_MASKS[from_sq]]
                 | chess.BB_FILE_ATTACKS[from_sq][occ & chess.BB_FILE_MASKS[from_sq]])
                & to_bb
            )

        return bool(
            (chess.BB_DIAG_ATTACKS[from_sq][occ & chess.BB_DIAG_MASKS[from_sq]]
             | chess.BB_RANK_ATTACKS[from_sq][occ & chess.BB_RANK_MASKS[from_sq]]
             | chess.BB_FILE_ATTACKS[from_sq][occ & chess.BB_FILE_MASKS[from_sq]])
            & to_bb
        )


    def _lva_sq(self, color: bool, to_sq: int, occ: int):
        """
        Least-valuable attacker of `color` on `to_sq` given occupancy `occ`.

        Pieces are tried in ascending value order so the first hit is the LVA.
        Absolutely-pinned pieces are skipped — they cannot legally move.

        Returns the square index of the LVA, or None.
        """
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING):
            candidates = self.board.pieces(piece_type, color) & occ
            if not candidates:
                continue

            for sq in chess.SquareSet(candidates):
                if not self._attacks_to_sq(piece_type, color, sq, to_sq, occ):
                    continue

                if self.board.is_pinned(color, sq):
                    pin_ray = self.board.pin(color, sq)
                    if not (pin_ray & chess.BB_SQUARES[to_sq]):
                        continue

                return sq

        return None


    def see(self, move: chess.Move) -> int:

        to_sq   = move.to_square
        from_sq = move.from_square

        if not self.board.is_legal(move):
            return 0

        target   = self.board.piece_at(to_sq)
        attacker = self.board.piece_at(from_sq)
        if target is None or attacker is None:
            return 0


        attacked_values = [self.pieces_values[target.piece_type]]

        occ = self.board.occupied ^ chess.BB_SQUARES[from_sq]

        val_on_sq = self.pieces_values.get(move.promotion) or self.pieces_values[attacker.piece_type]

        color = not attacker.color

        while True:
            lva_sq = self._lva_sq(color, to_sq, occ)
            if lva_sq is None:
                break

            lva = self.board.piece_at(lva_sq)

            attacked_values.append(val_on_sq)
            val_on_sq = self.pieces_values[lva.piece_type]
            occ      ^= chess.BB_SQUARES[lva_sq]
            color     = not color


        gain = attacked_values[-1]
        for i in range(len(attacked_values) - 2, -1, -1):
            gain = attacked_values[i] - max(0, gain)

        return gain



# import cProfile
# import pstats

# board = chess.Board()
# def benchmark():
#     searcher = Searcher(board, Handcrafted())
#     print(searcher.search(8))

# profiler = cProfile.Profile()
# profiler.enable()

# benchmark()

# profiler.disable()

# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime")
# stats.print_stats(25)
