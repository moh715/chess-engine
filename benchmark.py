import time

import chess

from player import Searcher, evaluate

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
