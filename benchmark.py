import time

import chess

import random
from player import Searcher
from evaluation import evaluate, evaluate2

board = chess.Board()

searcher = Searcher(board, evaluate)

start = time.perf_counter()

score, move = searcher.search(8)

elapsed = time.perf_counter() - start

print("Best move:", move)
print("Score:", score)
print("Time:", round(elapsed, 3), "seconds")
print("Nodes:", searcher.nodes)
print("NPS:", int(searcher.nodes / elapsed))

print("TT lookups:", searcher.tt_lookups)
print("TT hits:", searcher.tt_hits)

if searcher.tt_lookups:
    print("TT hit rate:", round(100 * searcher.tt_hits / searcher.tt_lookups, 2), "%")
# pieces = {
#         chess.PAWN,
#         chess.ROOK,
#         chess.KNIGHT,
#         chess.BISHOP,
#         chess.QUEEN,
#         chess.KING
#     }
# GAMES = 20
# DEPTH = 5

# eval1_wins = 0
# eval2_wins = 0
# draws = 0

# def pieces_count(board):
#     white = {}
#     black = {}
#     for p in pieces:
        
#         white[p] =   len(board.pieces(p, chess.WHITE))
#         black[p] = len(board.pieces(p, chess.BLACK))
#     return white, black

    
# def random_position(max_plies=20):
#     board = chess.Board()

#     n = random.randint(8, max_plies)

#     for _ in range(n):
#         if board.is_game_over():
#             break

#         move = random.choice(list(board.legal_moves))
#         board.push(move)

#     return board
# test_pos = [chess.Board("q7/1k6/8/8/8/8/8/4K3 w - - 0 1")]
# for game_num in range(1):
#     board = test_pos[game_num]

#     # Alternate colors
#     if game_num % 2 == 0:
#         white_eval = evaluate
#         black_eval = evaluate
#         eval1_is_white = True
#     else:
#         white_eval = evaluate
#         black_eval = evaluate
#         eval1_is_white = False

#     move_count = 0
#     white_searcher = Searcher(board, white_eval) 
#     black_searcher = Searcher(board, black_eval)
#     while not board.is_game_over():
#         if board.turn == chess.WHITE:
#             _, move = white_searcher.search(DEPTH)
#         else:
#             _, move = black_searcher.search(DEPTH)
#         if move is None:
#             break

#         board.push(move)
#         move_count += 1

#     outcome = board.outcome()

#     if outcome is None or outcome.winner is None:
#         draws += 1
        
#         if outcome is None:
#             print("Unknown result")
#         elif outcome.winner is None:
#             print("Draw:", outcome.termination)
#             white, black = pieces_count(board)
#             eval1_count = white if eval1_is_white else black
#             eval2_count = black if eval1_is_white else white
#             print(f"eval1:{eval1_count}")
#             print(f"eval2:{eval2_count}")
#             print(f"fen: {board.fen()}")
#             print(f"white is eval1:{eval1_is_white}")
#             turn = "white" if board.turn == chess.WHITE else "black"
#             print(f"turn: {turn}")
#             print(f"eval1: {evaluate(board)}")
#             print(f"eval2: {evaluate2(board)}")
            
#     elif outcome.winner == chess.WHITE:
#         if eval1_is_white:
#             eval1_wins += 1
#         else:
#             eval2_wins += 1

#     else:  # Black won
#         if eval1_is_white:
#             eval2_wins += 1
#         else:
#             eval1_wins += 1

#     print(
#         f"Game {game_num + 1}/{GAMES} | "
#         f"Eval1: {eval1_wins}  "
#         f"Eval2: {eval2_wins}  "
#         f"Draws: {draws}"
#     )

# print("\nFINAL RESULTS")
# print("Eval1 wins:", eval1_wins)
# print("Eval2 wins:", eval2_wins)
# print("Draws:", draws)

# total_decisive = eval1_wins + eval2_wins
# if total_decisive:
#     print(
#         "Eval1 score:",
#         round((eval1_wins + draws * 0.5) / GAMES * 100, 1),
#         "%"
#     )
#     print(
#         "Eval2 score:",
#         round((eval2_wins + draws * 0.5) / GAMES * 100, 1),
#         "%"
#     )