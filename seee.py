import bulletchess as bc
from bulletchess.utils import is_pinned

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

    DIAG_DIRS = ["ne", "nw", "se", "sw"]
    ORTHO_DIRS = ["north", "south", "east", "west"]

    PIECE_VALUES = {
        bc.PAWN: 100,
        bc.KNIGHT: 320,
        bc.BISHOP: 330,
        bc.ROOK: 500,
        bc.QUEEN: 900,
        bc.KING: 2000,
    }

    def __init__(self, board: bc.Board, piece_values: dict = None):
        self.board = board
        self.pieces_values = piece_values or self.PIECE_VALUES

        
    def _ray(self, from_sq: bc.Square, direction: str, occ: bc.Bitboard) -> bc.Bitboard:
        """Walk one direction from from_sq, stopping after (and including) the first occupied square."""
        attacked = bc.EMPTY_BB
        sq = getattr(from_sq, direction)()
        while sq is not None:
            attacked |= sq
            if sq in occ:
                break
            sq = getattr(sq, direction)()
        return attacked


    def _attacks_to_sq(
        self,
        piece_type,
        color: bc.Color,
        from_sq: bc.Square,
        to_sq: bc.Square,
        occ: bc.Bitboard,
    ) -> bool:
        if piece_type == bc.PAWN:
            targets = (
                (from_sq.ne(), from_sq.nw())
                if color == bc.WHITE
                else (from_sq.se(), from_sq.sw())
            )
            return to_sq in [t for t in targets if t is not None]

        if piece_type == bc.KNIGHT:
            targets = [step(from_sq) for step in self.KNIGHT_STEPS]
            return to_sq in [t for t in targets if t is not None]

        if piece_type == bc.KING:
            return to_sq in from_sq.adjacent()

        if piece_type == bc.BISHOP:
            attacked = bc.EMPTY_BB
            for d in self.DIAG_DIRS:
                attacked |= self._ray(from_sq, d, occ)
            return to_sq in attacked

        if piece_type == bc.ROOK:
            attacked = bc.EMPTY_BB
            for d in self.ORTHO_DIRS:
                attacked |= self._ray(from_sq, d, occ)
            return to_sq in attacked

        attacked = bc.EMPTY_BB
        for d in self.DIAG_DIRS + self.ORTHO_DIRS:
            attacked |= self._ray(from_sq, d, occ)
        return to_sq in attacked
    def _lva_sq(self, color: bc.Color, to_sq: bc.Square, occ: bc.Bitboard):
        for piece_type in (bc.PAWN, bc.KNIGHT, bc.BISHOP, bc.ROOK, bc.QUEEN, bc.KING):
            candidates = self.board[color, piece_type] & occ
            if not candidates:
                continue
            for sq in candidates:
                if not self._attacks_to_sq(piece_type, color, sq, to_sq, occ):
                    continue
                if is_pinned(self.board, sq):
                    continue
                return sq
        return None


    def see(self, move: bc.Move) -> int:
        to_sq = move.origin
        from_sq = move.destination

        if move not in self.board.legal_moves():
            return 0

        target = self.board[to_sq]
        attacker = self.board[from_sq]
        if target is None or attacker is None:
            return 0

        attacked_values = [self.pieces_values[target.piece_type]]
        occ = (self.board[bc.WHITE] | self.board[bc.BLACK]) ^ from_sq
        val_on_sq = (
            self.pieces_values.get(move.promotion)
            or self.pieces_values[attacker.piece_type]
        )
        color = attacker.color.opposite

        while True:
            lva_sq = self._lva_sq(color, to_sq, occ)
            if lva_sq is None:
                break
            lva = self.board[lva_sq]
            attacked_values.append(val_on_sq)
            val_on_sq = self.pieces_values[lva.piece_type]
            occ ^= lva_sq
            color = color.opposite

        gain = attacked_values[-1]
        for i in range(len(attacked_values) - 2, -1, -1):
            gain = attacked_values[i] - max(0, gain)
        return gain


if __name__ == "__main__":
    board = bc.Board.from_fen("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
    move = bc.Move(bc.C4, bc.D5)  # pawn takes pawn
    evaluator = SEEEvaluator(board)
    print(evaluator.see(move))  # expect 100 (wins a pawn, nothing recaptures)
