import bulletchess as bc



class SEEEvaluator:
    PIECE_VALUES = {
        bc.PAWN: 100,
        bc.ROOK: 500,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.QUEEN: 900,
        bc.KING: 2000,
    }
 
    def __init__(self, board: bc.Board, piece_values: dict = None):
        self.board = board
        self.pieces_values = piece_values or self.PIECE_VALUES
        self._cache = {}
        self.hits = 0
 
    def see(self, move: bc.Move, assume_legal: bool = False) -> int:
        key = (self.board, move)
        if key in self._cache:
            return self._cache[key]
        if not move.is_capture(self.board):
            self._cache[key] = 0
            return 0
        gain = self.pieces_values[self.board[move.destination].piece_type]
        us = self.board.turn
        played = 1
        self.board.apply(move)
        while True:
            moves = [m for m in self.board.legal_moves() if m.destination == move.destination]
            if not moves:
                break
            played += 1
            next_move = min(moves, key=lambda m: self.pieces_values[self.board[m.origin].piece_type])
            gain += self.pieces_values[self.board[next_move.origin].piece_type] if self.board.turn == us else -self.pieces_values[self.board[next_move.origin].piece_type]
            self.board.apply(next_move)
        for _ in range(played):    
            self.board.undo()
        self._cache[key] = gain
        return gain

        
if __name__ == "__main__":
    board = bc.Board.from_fen("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
    move = bc.Move(bc.C4, bc.D5)  
    evaluator = SEEEvaluator(board)
    print(evaluator.see(move))  
    for sq in bc.A1.bb():
        print(sq)
