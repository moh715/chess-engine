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
 
    def see(self, move: bc.Move) -> int:
            if not move.is_capture(self.board):
                return 0
                
            board = self.board.copy()  # never mutate the live search board
            
            # 1. Determine the value of the initially captured piece
            captured = board[move.destination]
            if captured is None:
                # en passant: captured pawn sits beside the destination, not on it
                captured_square = move.destination.south() if board.turn == bc.WHITE else move.destination.north()
                captured = board[captured_square]
                
            gain = self.pieces_values[captured.piece_type]
            us = self.board.turn
            
            board.apply(move)
            
            while True:
                moves = [m for m in board.legal_moves() if m.destination == move.destination]
                if not moves:
                    break
                    
                next_move = min(moves, key=lambda m: self.pieces_values[board[m.origin].piece_type])
                
                captured_val = self.pieces_values[board[move.destination].piece_type]
                
                if board.turn == us:
                    gain += captured_val
                else:
                    gain -= captured_val
                    
                board.apply(next_move)
                
            return gain

        
if __name__ == "__main__":
    board = bc.Board.from_fen("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
    move = bc.Move(bc.C4, bc.D5)  
    evaluator = SEEEvaluator(board)
    print(evaluator.see(move))  
    for sq in bc.A1.bb():
        print(sq)
