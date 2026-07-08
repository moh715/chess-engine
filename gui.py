import pygame
import chess
from player import Searcher
from evaluation import evaluate, evaluate2

# -------------------------
# Your Searcher class here
# -------------------------

# -------------------------
# Your evaluate() here
# -------------------------

pygame.init()

SIZE = 640
SQ_SIZE = SIZE // 8

screen = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption("Play Against Your Engine")

font = pygame.font.SysFont("segoeuisymbol", 60)

board = chess.Board()
depth = 3
searcher = Searcher(board, evaluate2)

selected = None

pieces = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
}


def draw():
    colors = [(240, 217, 181), (181, 136, 99)]

    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]

            pygame.draw.rect(
                screen,
                color,
                (file * SQ_SIZE, rank * SQ_SIZE,
                 SQ_SIZE, SQ_SIZE)
            )

            square = chess.square(file, 7 - rank)

            piece = board.piece_at(square)

            if piece:
                text = font.render(
                    pieces[piece.symbol()],
                    True,
                    (0, 0, 0)
                )

                rect = text.get_rect(
                    center=(
                        file * SQ_SIZE + SQ_SIZE // 2,
                        rank * SQ_SIZE + SQ_SIZE // 2
                    )
                )

                screen.blit(text, rect)

    if selected is not None:
        file = chess.square_file(selected)
        rank = 7 - chess.square_rank(selected)

        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (file * SQ_SIZE,
             rank * SQ_SIZE,
             SQ_SIZE,
             SQ_SIZE),
            4
        )

    pygame.display.flip()


running = True

while running:

    draw()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:

            x, y = pygame.mouse.get_pos()

            file = x // SQ_SIZE
            rank = 7 - (y // SQ_SIZE)

            square = chess.square(file, rank)

            if selected is None:

                piece = board.piece_at(square)

                if piece and piece.color == chess.WHITE:
                    selected = square

            else:

                move = chess.Move(selected, square)

                if move not in board.legal_moves:

                    promo = chess.Move(
                        selected,
                        square,
                        promotion=chess.QUEEN
                    )

                    if promo in board.legal_moves:
                        move = promo
                    else:
                        selected = None
                        continue

                board.push(move)

                selected = None

                draw()

                if not board.is_game_over():

                    print("Engine thinking...")

                    score, pv = searcher.search(depth)

                    print("Score:", score)
                    print("PV:", pv)

                    if pv:
                        board.push(pv)
                    else:
                        print(board.fen())
                if board.is_game_over():
                    running = False

pygame.quit()