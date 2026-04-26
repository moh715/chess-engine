import chess


def minmax(board:chess.Board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)
    if board.is_checkmate():
        return float("-inf")
    best_value = float("-inf")
    for move in board.legal_moves():
        newscore = -minmax(board.push(move), depth-1, -beta, -alpha)
        board.pop()
        if newscore > best_value:
            best_value = newscore
            if best_value > alpha:
                alpha = best_value
        if newscore >= beta:
            return newscore
    return best_value
               


            
def evaluate(board:chess.Board):
    pass

    

