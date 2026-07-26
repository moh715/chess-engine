import bulletchess as bc



class SEEEvaluator:
    """
    Static Exchange Evaluation for bulletchess.
 
    Usage:
        evaluator = SEEEvaluator(board)
        score = evaluator.see(move)
    """
 
    KNIGHT_STEPS = [
        lambda s: s.north(2) and s.north(2).east(1),
        lambda s: s.north(2) and s.north(2).west(1),
        lambda s: s.south(2) and s.south(2).east(1),
        lambda s: s.south(2) and s.south(2).west(1),
        lambda s: s.east(2) and s.east(2).north(1),
        lambda s: s.east(2) and s.east(2).south(1),
        lambda s: s.west(2) and s.west(2).north(1),
        lambda s: s.west(2) and s.west(2).south(1),
    ]
 
    DIAG_DIRS = (
        bc.Square.ne,
        bc.Square.nw,
        bc.Square.se,
        bc.Square.sw,
    )
    ORTHO_DIRS = (bc.Square.north, bc.Square.south, bc.Square.east, bc.Square.west)
 
    PIECE_VALUES = {
        bc.PAWN: 100,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.ROOK: 500,
        bc.QUEEN: 900,
        bc.KING: 0,
    }
 
    RAYS = {}
 
    for step in DIAG_DIRS + ORTHO_DIRS:
        table = {}
 
        for start in bc.SQUARES:
            squares = []
 
            sq = step(start)
            while sq is not None:
                squares.append(sq)
                sq = step(sq)
 
            table[start] = bc.Bitboard(squares)
 
        RAYS[step] = table
 
    KNIGHT_ATTACKS = {}
    KING_ATTACKS = {}
    WHITE_PAWN_ATTACKS = {}
    BLACK_PAWN_ATTACKS = {}
 
    for sq in bc.SQUARES:
        knight = []
        for step in KNIGHT_STEPS:
            dst = step(sq)
            if dst is not None:
                knight.append(dst)
        KNIGHT_ATTACKS[sq] = bc.Bitboard(knight)
 
        KING_ATTACKS[sq] = bc.Bitboard(sq.adjacent())
 
        WHITE_PAWN_ATTACKS[sq] = bc.Bitboard(
            [x for x in (sq.ne(), sq.nw()) if x is not None]
        )
 
        BLACK_PAWN_ATTACKS[sq] = bc.Bitboard(
            [x for x in (sq.se(), sq.sw()) if x is not None]
        )
 
    def __init__(self, board: bc.Board, piece_values: dict = None):
        self.board = board
        self.pieces_values = piece_values or self.PIECE_VALUES
        self._cache = {}
        self.hits = 0
 
    def _ray_blocker(self, from_sq, step, occ):
        """
        Walk a ray from `from_sq` in direction `step` and return the first
        square in `occ` (or None if the ray runs off the board unblocked).
        """
        for sq in self.RAYS[step][from_sq]:
            if sq in occ:
                return sq
        return None
 
    def _attackers(self, to_sq: bc.Square, occ: bc.Bitboard, piece_bbs) -> bc.Bitboard:
        """
        Bitboard of every piece (either color) currently in `occ` that
        attacks `to_sq`. Instead of testing every piece on the board, this
        walks the 8 rays outward from `to_sq` (plus O(1) knight/king/pawn
        lookups) and grabs the first blocker in each direction - the only
        candidate that could possibly be attacking `to_sq` along that line.
        """
        knights, kings, bishops_queens, rooks_queens, white_pawns, black_pawns = piece_bbs
 
        attackers = bc.EMPTY_BB
        attackers |= self.KNIGHT_ATTACKS[to_sq] & knights & occ
        attackers |= self.KING_ATTACKS[to_sq] & kings & occ
 
       
        attackers |= self.BLACK_PAWN_ATTACKS[to_sq] & white_pawns & occ
        attackers |= self.WHITE_PAWN_ATTACKS[to_sq] & black_pawns & occ
 
        for d in self.DIAG_DIRS:
            blocker = self._ray_blocker(to_sq, d, occ)
            if blocker is not None and blocker in bishops_queens:
                attackers |= blocker
 
        for d in self.ORTHO_DIRS:
            blocker = self._ray_blocker(to_sq, d, occ)
            if blocker is not None and blocker in rooks_queens:
                attackers |= blocker
 
        return attackers
 
    def _pinned_in_occ(
        self, sq: bc.Square, color: bc.Color, occ: bc.Bitboard, king_sq: bc.Square
    ) -> bool:
        """
        Is the piece on `sq` pinned against its king given the *current*
        (possibly reduced) occupancy `occ`, rather than the live board?
 
        Mid-exchange, pieces get virtually removed from `occ`. That can
        create a pin that doesn't exist on the real board (the blocker in
        front of `sq` was just "captured" earlier in the sequence) or
        remove one that does exist there (the pinning slider itself was
        already "captured"). This checks directly against `occ` so it
        stays correct as the exchange progresses.
        """
        if king_sq is None:
            return False
 
        direction = None
        for d in self.DIAG_DIRS + self.ORTHO_DIRS:
            if sq in self.RAYS[d][king_sq]:
                direction = d
                break
        if direction is None:
            return False
 
        if self._ray_blocker(king_sq, direction, occ) != sq:
            return False
 
        beyond = self._ray_blocker(king_sq, direction, occ ^ sq)
        if beyond is None:
            return False
        piece = self.board[beyond]
        if piece is None or piece.color == color:
            return False
 
        if direction in self.DIAG_DIRS:
            return piece.piece_type in (bc.BISHOP, bc.QUEEN)
        return piece.piece_type in (bc.ROOK, bc.QUEEN)
 
    def _lva_sq(
        self, color: bc.Color, to_sq: bc.Square, occ: bc.Bitboard, piece_bbs, king_sq: bc.Square
    ):
        attackers = self._attackers(to_sq, occ, piece_bbs) & self.board[color]
        if not attackers:
            return None
 
        for piece_type in (bc.PAWN, bc.KNIGHT, bc.BISHOP, bc.ROOK, bc.QUEEN, bc.KING):
            for sq in attackers & self.board[color, piece_type]:
                if self._pinned_in_occ(sq, color, occ, king_sq):
                    continue
                return sq
 
        return None
 
    def see(self, move: bc.Move, assume_legal: bool = False) -> int:
        """
        Compute the Static Exchange Evaluation score for `move`.
 
        `assume_legal`: set True if `move` is already known to be legal
        (e.g. it came from `board.legal_moves()`) to skip re-generating the
        full legal move list on every call - that generation is by far the
        most expensive part of this function when called repeatedly (e.g.
        during move ordering).
        """
        if not move.is_capture(self.board):
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
        return gain
if __name__ == "__main__":
    board = bc.Board.from_fen("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
    move = bc.Move(bc.C4, bc.D5)  
    evaluator = SEEEvaluator(board)
    print(evaluator.see(move))  
    for sq in bc.A1.bb():
        print(sq)
