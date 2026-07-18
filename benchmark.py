import random
import time

import chess

from evaluation import Handcrafted, NNEvaluation
from searcher import Searcher

ran = random.Random(42)

def random_position(max_plies=20):
    board = chess.Board()

    n = ran.randint(8, max_plies)

    for _ in range(n):
        if board.is_game_over():
            break

        move = ran.choice(list(board.legal_moves))
        board.push(move)

    return board

    
def single_move_benchmark(depth, eval):
    board = chess.Board("2b1kbnr/4r2p/p1Rp2pq/1P2pp2/2BPPBP1/2Pn4/1PK1NP1R/3Q2N1 w k - 4 18")

    searcher = Searcher(board, eval())

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

        searcher = Searcher(board, eval())

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
multi_fen_benchmark(3, Handcrafted)
# Total time   : 47.562 s
# Average time : 0.951 s

# Totals
# Nodes        : 614,333
# TT hits      : 27,190
# TT lookups   : 614,286
# Beta cutoffs : 189,043
# Aspiration fails : 30
# Sort time    : 0.044 s
# Eval time    : 7.897 s
# Quiesce time : 45.981 s
# TT time      : 0.641 s
# NPS: 12917
# TT hit rate: 4.43%
# Beta cutoff rate: 30.77%

# Averages
# Nodes        : 12286.7
# TT hits      : 543.8
# TT lookups   : 12285.7
# Beta cutoffs : 3780.9
# Aspiration fails : 0.60
# Sort time    : 0.0009 s
# Eval time    : 0.1579 s
# Quiesce time : 0.9196 s
# TT time      : 0.0128 s

# Moves: [<Move: c5b6>, <Move: a4b5>, <Move: b4b5>, <Move: c4d3>, <Move: e3f1>, <Move: a4e4>, <Move: g3g5>, <Move: b6b3>, <Move: f1e2>, <Move: c5d6>, <Move: h6g5>, <Move: f1e2>, <Move: g4h3>, <Move: b3b1>, <Move: c5b4>, <Move: g1f3>, <Move: e5f3>, <Move: c2b3>, <Move: b1c3>, <Move: c3a4>, <Move: g4g3>, <Move: f1e2>, <Move: b5a4>, <Move: d4c2>, <Move: g8f6>, <Move: f4d6>, <Move: f6h4>, <Move: c1b2>, <Move: c4d3>, <Move: a6b6>, <Move: g2f3>, <Move: b7h1>, <Move: d7d6>, <Move: d8d4>, <Move: c1a3>, <Move: d6e5>, <Move: e6g7>, <Move: b6c5>, <Move: b4c2>, <Move: b5c4>, <Move: d1e1>, <Move: f6d5>, <Move: b4c5>, <Move: b5c4>, <Move: f6f8>, <Move: g8f6>, <Move: c8g4>, <Move: h6f7>, <Move: g1h3>, <Move: e6f5>]