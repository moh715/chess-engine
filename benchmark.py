import random
import time

import chess

from evaluation import Handcrafted, NNEvaluation
from searcher import Searcher
import bulletchess as bc

ran = random.Random(42)

def random_position(max_plies=20):
    board = bc.Board()

    n = ran.randint(8, max_plies)

    for _ in range(n):
        if board in bc.MATE:
            break

        move = ran.choice(list(board.legal_moves()))
        board.apply(move)

    return board

    
def single_move_benchmark(depth, eval):
    board = bc.Board.from_fen("2b1kbnr/4r2p/p1Rp2pq/1P2pp2/2BPPBP1/2Pn4/1PK1NP1R/3Q2N1 w k - 4 18")

    searcher = Searcher(board, eval)

    start = time.perf_counter()

    score, move = searcher.search(depth)

    elapsed = time.perf_counter() - start

    print("Best move:", move)
    print("Score:", score)
    print("Time:", round(elapsed, 3), "seconds")
    searcher.print_profile()

    if searcher.tt_lookups:
        print("TT hit rate:", round(100 * searcher.tt_hits / searcher.tt_lookups, 2), "%")


def game_benchmark(depth, eval):
    board = chess.Board()
    searcher = Searcher(board, eval())
    start = time.perf_counter()
    moves = 0
    while not board.is_game_over():
        moves += 1
        _, move = searcher.search(depth)
        board.push(move)
    print("time: ", round(time.perf_counter() - start, 3), "s")
    print(f"moves: {moves}")
    searcher.print_profile()
    
def multi_fen_benchmark(depth, eval):
    test_fens = [random_position(50) for _ in range(50)]

    total_time = 0.0
    moves = []
    scores = []

    totals = {
        "nodes": 0,
        "tt_hits": 0,
        "tt_lookups": 0,
        "beta_cutof": 0,
        "aspr_fail": 0,
        "time_sort": 0.0,
        "time_eval": 0.0,
        "time_quiesce": 0.0,
        "time_tt": 0.0,
    }

    for board in test_fens:
        print("fen:", board.fen())

        searcher = Searcher(board, eval)

        start = time.perf_counter()
        score, move = searcher.search(depth)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        moves.append(move)
        scores.append(score)

        totals["nodes"] += searcher.nodes
        totals["tt_hits"] += searcher.tt_hits
        totals["tt_lookups"] += searcher.tt_lookups
        totals["beta_cutof"] += searcher.beta_cutof
        totals["aspr_fail"] += searcher.aspr_fail
        totals["time_sort"] += searcher.time_sort
        totals["time_eval"] += searcher.time_eval
        totals["time_quiesce"] += searcher.time_quiesce
        totals["time_tt"] += searcher.time_tt

        print(
            f"Best move: {move}  "
            f"Score: {score}  "
            f"Time: {elapsed:.3f} seconds"
        )

    n = len(test_fens)

    print("=" * 50)
    print(f"Total time   : {total_time:.3f} s")
    print(f"Average time : {total_time / n:.3f} s")
    print()

    print("Totals")
    print(f"Nodes        : {totals['nodes']:,}")
    print(f"TT hits      : {totals['tt_hits']:,}")
    print(f"TT lookups   : {totals['tt_lookups']:,}")
    print(f"Beta cutoffs : {totals['beta_cutof']:,}")
    print(f"Aspiration fails : {totals['aspr_fail']:,}")
    print(f"Sort time    : {totals['time_sort']:.3f} s")
    print(f"Eval time    : {totals['time_eval']:.3f} s")
    print(f"Quiesce time : {totals['time_quiesce']:.3f} s")
    print(f"TT time      : {totals['time_tt']:.3f} s")
    print(f"NPS: {totals['nodes'] / total_time:.0f}")
    print(f"TT hit rate: {100 * totals['tt_hits'] / totals['tt_lookups']:.2f}%")
    print(f"Beta cutoff rate: {100 * totals['beta_cutof'] / totals['nodes']:.2f}%")
    print()
    print("Averages")
    print(f"Nodes        : {totals['nodes'] / n:.1f}")
    print(f"TT hits      : {totals['tt_hits'] / n:.1f}")
    print(f"TT lookups   : {totals['tt_lookups'] / n:.1f}")
    print(f"Beta cutoffs : {totals['beta_cutof'] / n:.1f}")
    print(f"Aspiration fails : {totals['aspr_fail'] / n:.2f}")
    print(f"Sort time    : {totals['time_sort'] / n:.4f} s")
    print(f"Eval time    : {totals['time_eval'] / n:.4f} s")
    print(f"Quiesce time : {totals['time_quiesce'] / n:.4f} s")
    print(f"TT time      : {totals['time_tt'] / n:.4f} s")

    print()
    print("Moves:", moves)

def eval_vs_eval2(depth, games, eval_fns=[Handcrafted, NNEvaluation]):
    pieces ={
            chess.PAWN,
            chess.ROOK,
            chess.KNIGHT,
            chess.BISHOP,
            chess.QUEEN,
            chess.KING
        }
    
    eval1_wins = 0
    eval2_wins = 0
    draws = 0
    
    def pieces_count(board):
        white = {}
        black = {}
        for p in pieces:
    
            white[p] =   len(board.pieces(p, chess.WHITE))
            black[p] = len(board.pieces(p, chess.BLACK))
        return white, black
    
    
    test_pos = [random_position() for _ in range(games)]
    for game_num in range(games):
        board = test_pos[game_num]
    
        # Alternate colors
        if game_num % 2 == 0:
            white_eval = eval_fns[0]()
            black_eval = eval_fns[1]() 
            eval1_is_white = True
        else:
            white_eval = eval_fns[1]()
            black_eval = eval_fns[0]()
            eval1_is_white = False
    
        move_count = 0
        white_searcher = Searcher(board, white_eval)
        black_searcher = Searcher(board, black_eval)
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                _, move = white_searcher.search(depth)
            else:
                _, move = black_searcher.search(depth)
            if move is None:
                break
    
            board.push(move)
            move_count += 1
    
        outcome = board.outcome()
    
        if outcome is None or outcome.winner is None:
            draws += 1
    
            if outcome is None:
                print("Unknown result")
            elif outcome.winner is None:
                print("Draw:", outcome.termination)
                white, black = pieces_count(board)
                eval1_count = white if eval1_is_white else black
                eval2_count = black if eval1_is_white else white
                print(f"eval1:{eval1_count}")
                print(f"eval2:{eval2_count}")
                print(f"fen: {board.fen()}")
                print(f"white is eval1:{eval1_is_white}")
                turn = "white" if board.turn == chess.WHITE else "black"
                print(f"turn: {turn}")
                print(f"eval1: {evaluate(board)}")
                print(f"eval2: {evaluate2(board)}")
    
        elif outcome.winner == chess.WHITE:
            if eval1_is_white:
                eval1_wins += 1
            else:
                eval2_wins += 1
    
        else:  # Black won
            if eval1_is_white:
                eval2_wins += 1
            else:
                eval1_wins += 1
    
        print(
            f"Game {game_num + 1}/{games} | "
            f"Eval1: {eval1_wins}  "
            f"Eval2: {eval2_wins}  "
            f"Draws: {draws}"
        )
    
    print("\nFINAL RESULTS")
    print("Eval1 wins:", eval1_wins)
    print("Eval2 wins:", eval2_wins)
    print("Draws:", draws)
    
    total_decisive = eval1_wins + eval2_wins
    if total_decisive:
        print(
            "Eval1 score:",
            round((eval1_wins + draws * 0.5) / games * 100, 1),
            "%"
        )
        print(
            "Eval2 score:",
            round((eval2_wins + draws * 0.5) / games * 100, 1),
            "%"
        )
# eval_vs_eval2(2, 10)
# fen: rn1qk1nr/p1pppp1p/7b/P5p1/1p5P/1P6/N1PPbPP1/R1BQKBNR w KQkq - 0 7
multi_fen_benchmark(4, Handcrafted)
# Total time   : 547.353 s
# Average time : 10.947 s

# Totals
# Nodes        : 6,469,859
# TT hits      : 236,467
# TT lookups   : 6,471,913
# Beta cutoffs : 2,153,958
# Aspiration fails : 74
# Sort time    : 0.072 s
# Eval time    : 73.165 s
# Quiesce time : 536.030 s
# TT time      : 7.707 s
# NPS: 11820
# TT hit rate: 3.65%
# Beta cutoff rate: 33.29%

# Averages
# Nodes        : 129397.2
# TT hits      : 4729.3
# TT lookups   : 129438.3
# Beta cutoffs : 43079.2
# Aspiration fails : 1.48
# Sort time    : 0.0014 s
# Eval time    : 1.4633 s
# Quiesce time : 10.7206 s
# TT time      : 0.1541 s
# fen: rn3bnr/p3p3/b1qpkp2/7B/P2PPPpP/2p1K1P1/1PPNNR2/R1BQ4 w - - 1 25
# Moves: [<Move: c5b6>, <Move: a4b5>, <Move: b4b5>, <Move: c4d3>, <Move: e3f1>, <Move: a4e4>, <Move: c5d6>, <Move: b6b3>, <Move: f1e2>, <Move: c5d6>, <Move: h6g5>, <Move: b1d2>, <Move: g4h3>, <Move: b5c6>, <Move: b5c4>, <Move: g1f3>, <Move: e5g4>, <Move: c2b3>, <Move: e2g3>, <Move: c3a4>, <Move: g4g3>, <Move: f1e2>, <Move: b5a4>, <Move: d4c2>, <Move: d8h4>, <Move: f4d6>, <Move: f6g5>, <Move: c1b2>, <Move: c4d3>, <Move: a6b6>, <Move: g2f3>, <Move: b7h1>, <Move: d7d6>, <Move: d8d4>, <Move: c1a3>, <Move: d6e5>, <Move: e6g7>, <Move: b6c5>, <Move: b4c2>, <Move: b5c4>, <Move: d1e1>, <Move: f6d5>, <Move: b4c5>, <Move: d5c4>, <Move: f6f8>, <Move: g8f6>, <Move: h6g4>, <Move: a4b5>, <Move: b4c3>, <Move: e6f5>]
=======
multi_fen_benchmark(4, Handcrafted())
>>>>>>> using-bullitchess
