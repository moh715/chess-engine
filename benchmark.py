from doctest import testsource
import random
import time

from tensorflow.keras.models import load_model
import cProfile
import pstats
from evaluation import Handcrafted, NNEvaluation
from searcher import Searcher
import bulletchess as bc

ran = random.Random(42)

def random_position(max_plies=40):
    board = bc.Board()

    n = ran.randint(10, max_plies)

    for _ in range(n):
        if board in bc.MATE:
            break

        move = ran.choice(list(board.legal_moves()))
        board.apply(move)

    return board

    
def single_move_benchmark(depth, eval):
    board = bc.Board.from_fen("2b1kbnr/4r2p/p1Rp2pq/1P2pp2/2BPPBP1/2Pn4/1PK1NP1R/3Q2N1 w k - 4 18")

    searcher = Searcher(board, eval)

    
    profiler = cProfile.Profile()
    profiler.enable()

    score, move = searcher.search(depth)

    profiler.disable()
        
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(25)
    if searcher.tt_lookups:
        print("TT hit rate:", round(100 * searcher.tt_hits / searcher.tt_lookups, 2), "%")
    print(f"score: {score} move:{move}")


def game_benchmark(depth, eval):
    board = bc.Board()
    searcher = Searcher(board, eval)
    start = time.perf_counter()
    moves = 0
    while board not in bc.MATE:
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

    print()
    print("Moves:", moves)


    
def eval_vs_eval2(depth, games, eval_fns=[Handcrafted, NNEvaluation]):
    pieces = {
        bc.PAWN,
        bc.ROOK,
        bc.KNIGHT,
        bc.BISHOP,
        bc.QUEEN,
        bc.KING
    }

    eval1_wins = 0
    eval2_wins = 0
    draws = 0

    def pieces_count(board):
        white = {}
        black = {}
        for p in pieces:
            white[p] = len(list(board[bc.WHITE, p]))
            black[p] = len(list(board[bc.BLACK, p]))
        return white, black
    test_pos = []
    test_pos.append(bc.Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    for _ in range(games - 1):
        test_pos.append(random_position())

    # Create the two searchers ONCE, tied to eval_fns[0] and eval_fns[1]
    # respectively. We'll rebind their board each game rather than
    # constructing new Searcher objects.
    searcher1 = Searcher(test_pos[0], eval_fns[0])
    searcher2 = Searcher(test_pos[0], eval_fns[1])

    for game_num in range(games):
        board = test_pos[game_num]
        print(f"Game {game_num + 1}/{board.fen()}")

        # Point both persistent searchers at this game's board
        searcher1.reset(board)
        searcher2.reset(board)

        # Alternate colors
        if game_num % 2 == 0:
            white_searcher = searcher1
            black_searcher = searcher2
            eval1_is_white = True
        else:
            white_searcher = searcher2
            black_searcher = searcher1
            eval1_is_white = False

        move_count = 0
        while board not in bc.MATE and board not in bc.DRAW:
            if board.turn == bc.WHITE:
                _, move = white_searcher.search(depth)
            else:
                _, move = black_searcher.search(depth)

            if move is None:
                print(f"no move found, breaking")
                break
            board.apply(move)
            move_count += 1

        if board in bc.DRAW:
            draws += 1
            white, black = pieces_count(board)
            eval1_count = white if eval1_is_white else black
            eval2_count = black if eval1_is_white else white
            print(f"eval1:{eval1_count}")
            print(f"eval2:{eval2_count}")
            print(f"fen: {board.fen()}")
            print(f"white is eval1:{eval1_is_white}")
            turn = "white" if board.turn == bc.WHITE else "black"
            print(f"turn: {turn}")

        elif board in bc.CHECKMATE:
            if board.turn == bc.BLACK:  # White delivered checkmate
                if eval1_is_white:
                    eval1_wins += 1
                else:
                    eval2_wins += 1
            else:  # Black delivered checkmate
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
# single_move_benchmark(7, Handcrafted)
eval_vs_eval2(5, 500, [NNEvaluation(), Handcrafted()])