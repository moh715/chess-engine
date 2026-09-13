"""
Chess engine testing GUI
-------------------------
- Choose to play White or Black
- Choose which evaluation function the engine uses
- Start from a custom FEN (or the normal start position)
- Copy the current board FEN to your clipboard at any time

Requires: pygame, bulletchess, and your own searcher.py / evaluation.py
(same interface as your original script: Searcher(board, eval_fn).search(depth) -> (score, move))
"""

import pygame
import bulletchess as bc
from searcher import Searcher
from evaluation import Handcrafted, NNEvaluation


def copy_to_clipboard(text):
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception as e:
        print("Could not copy to clipboard (", e, "). FEN:", text)
        return False


# ---------------------------------------------------------------------------
# Pygame setup
# ---------------------------------------------------------------------------
pygame.init()

BOARD_SIZE = 640
SQ_SIZE = BOARD_SIZE // 8
SIDEBAR_W = 300
WIN_W = BOARD_SIZE + SIDEBAR_W
WIN_H = BOARD_SIZE

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Chess Engine Tester")

piece_font = pygame.font.SysFont("segoeuisymbol", 60)
ui_font = pygame.font.SysFont("arial", 22)
ui_font_small = pygame.font.SysFont("arial", 17)
title_font = pygame.font.SysFont("arial", 34, bold=True)

WHITE_BG = (245, 245, 245)
LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
SEL_COLOR = (0, 200, 0)
LEGAL_DOT = (60, 60, 60)
BTN_COLOR = (70, 70, 90)
BTN_HOVER = (100, 100, 130)
BTN_SELECTED = (60, 140, 90)
TEXT_COLOR = (255, 255, 255)
DARK_TEXT = (20, 20, 20)
SIDEBAR_BG = (30, 30, 35)

# bulletchess doesn't expose Piece.symbol() the way python-chess does, so we
# build our own (piece_type, color) -> glyph lookup instead.
PIECE_GLYPHS = {
    (bc.PAWN,   bc.WHITE): "\u2659", (bc.PAWN,   bc.BLACK): "\u265F",
    (bc.KNIGHT, bc.WHITE): "\u2658", (bc.KNIGHT, bc.BLACK): "\u265E",
    (bc.BISHOP, bc.WHITE): "\u2657", (bc.BISHOP, bc.BLACK): "\u265D",
    (bc.ROOK,   bc.WHITE): "\u2656", (bc.ROOK,   bc.BLACK): "\u265C",
    (bc.QUEEN,  bc.WHITE): "\u2655", (bc.QUEEN,  bc.BLACK): "\u265B",
    (bc.KING,   bc.WHITE): "\u2654", (bc.KING,   bc.BLACK): "\u265A",
}

EVAL_FUNCS = {
    "handcrafted evaluation": Handcrafted,
    "NN Evaluation": NNEvaluation,
}


# ---------------------------------------------------------------------------
# Small UI widgets
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, rect, text, font=ui_font):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.selected = False

    def draw(self, surf):
        mouse = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse)
        if self.selected:
            color = BTN_SELECTED
        elif hovered:
            color = BTN_HOVER
        else:
            color = BTN_COLOR
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 1, border_radius=6)
        label = self.font.render(self.text, True, TEXT_COLOR)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class TextInput:
    def __init__(self, rect, placeholder=""):
        self.rect = pygame.Rect(rect)
        self.text = ""
        self.placeholder = placeholder
        self.active = False

    def draw(self, surf):
        color = (255, 255, 255) if self.active else (200, 200, 200)
        pygame.draw.rect(surf, (255, 255, 255), self.rect, border_radius=4)
        pygame.draw.rect(surf, color, self.rect, 2, border_radius=4)
        if self.text:
            label = ui_font_small.render(self.text, True, DARK_TEXT)
        else:
            label = ui_font_small.render(self.placeholder, True, (150, 150, 150))
        surf.blit(label, (self.rect.x + 6, self.rect.y + (self.rect.h - label.get_height()) // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
            else:
                self.text += event.unicode


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
class State:
    SETUP = "setup"
    PLAYING = "playing"


state = State.SETUP
player_color = bc.WHITE
eval_name = "NN Evaluation"
depth = 3
board = bc.Board()
searcher = None
selected = None          # bc.Square or None
legal_targets = []       # list of bc.Square
status_msg = ""
fen_error = ""

fen_input = TextInput((BOARD_SIZE // 2 - 220, 330, 440, 34),
                      "Paste a starting FEN here (optional)")

# Setup screen buttons
btn_white = Button((BOARD_SIZE // 2 - 220, 130, 200, 50), "Play White")
btn_black = Button((BOARD_SIZE // 2 + 20, 130, 200, 50), "Play Black")
btn_eval1 = Button((BOARD_SIZE // 2 - 220, 220, 200, 50), "handcrafted evaluation")
btn_eval2 = Button((BOARD_SIZE // 2 + 20, 220, 200, 50), "NN Evaluation")
btn_depth_minus = Button((BOARD_SIZE // 2 - 220, 400, 50, 40), "-")
btn_depth_plus = Button((BOARD_SIZE // 2 - 220 + 170, 400, 50, 40), "+")
btn_start = Button((BOARD_SIZE // 2 - 120, 470, 240, 55), "Start Game", font=title_font)

# In-game sidebar buttons (positions relative to sidebar, set up below)
btn_copy_fen = Button((BOARD_SIZE + 25, 20, SIDEBAR_W - 50, 45), "Copy FEN")
btn_new_game = Button((BOARD_SIZE + 25, 80, SIDEBAR_W - 50, 45), "New Game / Setup")
btn_flip = Button((BOARD_SIZE + 25, 140, SIDEBAR_W - 50, 45), "Flip Board View")

flipped_view = False  # only relevant if you want to manually flip; auto-set by color too


def sync_selected_buttons():
    btn_white.selected = (player_color == bc.WHITE)
    btn_black.selected = (player_color == bc.BLACK)
    btn_eval1.selected = (eval_name == "handcrafted evaluation")
    btn_eval2.selected = (eval_name == "NN Evaluation")


sync_selected_buttons()


def start_game():
    global board, searcher, state, selected, legal_targets, status_msg, fen_error, flipped_view
    fen_error = ""
    fen_text = fen_input.text.strip()
    if fen_text:
        try:
            board = bc.Board.from_fen(fen_text)
        except Exception:
            # bulletchess's from_fen error type isn't confirmed to be ValueError
            # specifically, so this is intentionally broad — narrow it back down
            # once you've verified what it actually raises on a bad FEN string.
            fen_error = "Invalid FEN, using standard start position instead."
            board = bc.Board()
    else:
        board = bc.Board()

    searcher = Searcher(board, EVAL_FUNCS[eval_name]())
    selected = None
    legal_targets = []
    status_msg = ""
    flipped_view = (player_color == bc.BLACK)
    state = State.PLAYING
    engine_move_if_needed()


def is_game_over(b: bc.Board) -> bool:
    """bulletchess has no single is_game_over(); combine the terminal statuses."""
    return b in bc.CHECKMATE or b in bc.DRAW


def engine_move_if_needed():
    """If it's the engine's turn, let it think and push its move."""
    global status_msg
    if is_game_over(board):
        return
    if board.turn != player_color:
        status_msg = "Engine is thinking..."
        draw()
        pygame.display.flip()
        score, pv = searcher.search(depth)
        print("Score:", score, "PV:", pv)
        if pv:
            board.apply(pv)
        else:
            print("No move returned. FEN:", board.fen())
        status_msg = ""


def board_result_text():
    if not is_game_over(board):
        return None
    if board in bc.CHECKMATE:
        winner = "Black" if board.turn == bc.WHITE else "White"
        return f"Checkmate — {winner} wins"
    if board in bc.STALEMATE:
        return "Draw — stalemate"
    if board in bc.INSUFFICIENT_MATERIAL:
        return "Draw — insufficient material"
    if board in bc.FIFTY_MOVE_TIMEOUT:
        return "Draw — fifty move rule"
    if board in bc.THREEFOLD_REPETITION:
        return "Draw — threefold repetition"
    return "Game over"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def screen_square_to_chess_square(file, rank):
    """Convert a screen file/rank (0..7, 0=top-left area) to a bulletchess Square,
    accounting for whether the board view is flipped."""
    if flipped_view:
        real_file = 7 - file
        real_rank = rank
    else:
        real_file = file
        real_rank = 7 - rank
    return bc.SQUARES[real_rank * 8 + real_file]


def chess_square_to_screen(square: bc.Square):
    idx = square.index()
    file = idx % 8
    rank = idx // 8
    if flipped_view:
        sx = 7 - file
        sy = rank
    else:
        sx = file
        sy = 7 - rank
    return sx, sy


def draw_setup():
    screen.fill(WHITE_BG)
    title = title_font.render("Chess Engine Tester", True, DARK_TEXT)
    screen.blit(title, title.get_rect(center=(BOARD_SIZE // 2, 60)))

    label = ui_font_small.render("Play as:", True, DARK_TEXT)
    screen.blit(label, (BOARD_SIZE // 2 - 220, 108))
    btn_white.draw(screen)
    btn_black.draw(screen)

    label = ui_font_small.render("Engine evaluation function:", True, DARK_TEXT)
    screen.blit(label, (BOARD_SIZE // 2 - 220, 198))
    btn_eval1.draw(screen)
    btn_eval2.draw(screen)

    label = ui_font_small.render("Engine search depth:", True, DARK_TEXT)
    screen.blit(label, (BOARD_SIZE // 2 - 220, 378))
    btn_depth_minus.draw(screen)
    btn_depth_plus.draw(screen)
    depth_label = ui_font.render(str(depth), True, DARK_TEXT)
    screen.blit(depth_label, depth_label.get_rect(
        center=(BOARD_SIZE // 2 - 220 + 110, 420)))

    label = ui_font_small.render("Starting position:", True, DARK_TEXT)
    screen.blit(label, (BOARD_SIZE // 2 - 220, 306))
    fen_input.draw(screen)

    if fen_error:
        err = ui_font_small.render(fen_error, True, (180, 0, 0))
        screen.blit(err, (BOARD_SIZE // 2 - 220, 368))

    btn_start.draw(screen)

    hint = ui_font_small.render(
        "Leave the FEN box empty to start from the normal position.",
        True, (100, 100, 100))
    screen.blit(hint, hint.get_rect(center=(BOARD_SIZE // 2, 545)))


def draw_board_area():
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQ if (rank + file) % 2 == 0 else DARK_SQ
            pygame.draw.rect(screen, color,
                              (file * SQ_SIZE, rank * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    for square in bc.SQUARES:
        piece = board[square]
        if piece is None:
            continue
        sx, sy = chess_square_to_screen(square)
        glyph = PIECE_GLYPHS[(piece.piece_type, piece.color)]
        text = piece_font.render(glyph, True, (0, 0, 0))
        rect = text.get_rect(center=(sx * SQ_SIZE + SQ_SIZE // 2,
                                      sy * SQ_SIZE + SQ_SIZE // 2))
        screen.blit(text, rect)

    if selected is not None:
        sx, sy = chess_square_to_screen(selected)
        pygame.draw.rect(screen, SEL_COLOR,
                          (sx * SQ_SIZE, sy * SQ_SIZE, SQ_SIZE, SQ_SIZE), 4)

    for target in legal_targets:
        sx, sy = chess_square_to_screen(target)
        center = (sx * SQ_SIZE + SQ_SIZE // 2, sy * SQ_SIZE + SQ_SIZE // 2)
        pygame.draw.circle(screen, LEGAL_DOT, center, 10)


def draw_sidebar():
    pygame.draw.rect(screen, SIDEBAR_BG, (BOARD_SIZE, 0, SIDEBAR_W, WIN_H))

    btn_copy_fen.draw(screen)
    btn_new_game.draw(screen)
    btn_flip.draw(screen)

    y = 200
    lines = [
        f"You are: {'White' if player_color == bc.WHITE else 'Black'}",
        f"Eval fn: {eval_name}",
        f"Depth: {depth}",
        f"Turn: {'White' if board.turn == bc.WHITE else 'Black'}",
    ]
    for line in lines:
        label = ui_font_small.render(line, True, TEXT_COLOR)
        screen.blit(label, (BOARD_SIZE + 25, y))
        y += 26

    y += 10
    if status_msg:
        label = ui_font_small.render(status_msg, True, (255, 210, 90))
        screen.blit(label, (BOARD_SIZE + 25, y))
        y += 30

    result = board_result_text()
    if result:
        label = ui_font_small.render(result, True, (120, 220, 120))
        screen.blit(label, (BOARD_SIZE + 25, y))
        y += 30

    # FEN text, wrapped
    y = WIN_H - 110
    label = ui_font_small.render("Current FEN:", True, TEXT_COLOR)
    screen.blit(label, (BOARD_SIZE + 25, y))
    y += 24
    fen_text = board.fen()
    max_chars = 34
    for i in range(0, len(fen_text), max_chars):
        chunk = fen_text[i:i + max_chars]
        label = ui_font_small.render(chunk, True, (200, 200, 200))
        screen.blit(label, (BOARD_SIZE + 25, y))
        y += 20


def draw():
    if state == State.SETUP:
        draw_setup()
    else:
        draw_board_area()
        draw_sidebar()
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
running = True
clock = pygame.time.Clock()

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        if state == State.SETUP:
            fen_input.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if btn_white.clicked(pos):
                    player_color = bc.WHITE
                elif btn_black.clicked(pos):
                    player_color = bc.BLACK
                elif btn_eval1.clicked(pos):
                    eval_name = "handcrafted evaluation"
                elif btn_eval2.clicked(pos):
                    eval_name = "NN Evaluation"
                elif btn_depth_minus.clicked(pos):
                    depth = max(1, depth - 1)
                elif btn_depth_plus.clicked(pos):
                    depth = min(8, depth + 1)
                elif btn_start.clicked(pos):
                    sync_selected_buttons()
                    start_game()
                sync_selected_buttons()

        elif state == State.PLAYING:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if btn_copy_fen.clicked(pos):
                    ok = copy_to_clipboard(board.fen())
                    status_msg = "FEN copied!" if ok else "Copy failed, see console."
                elif btn_new_game.clicked(pos):
                    state = State.SETUP
                    selected = None
                    legal_targets = []
                elif btn_flip.clicked(pos):
                    flipped_view = not flipped_view
                elif pos[0] < BOARD_SIZE and board.turn == player_color and not is_game_over(board):
                    file = pos[0] // SQ_SIZE
                    rank = pos[1] // SQ_SIZE
                    square = screen_square_to_chess_square(file, rank)
                    legal_moves = list(board.legal_moves())

                    if selected is None:
                        piece = board[square]
                        if piece and piece.color == player_color:
                            selected = square
                            legal_targets = [m.destination for m in legal_moves
                                              if m.origin == selected]
                    else:
                        if square == selected:
                            selected = None
                            legal_targets = []
                        else:
                            # bulletchess's Move() raises ValueError at construction time
                            # if the origin->destination shape is impossible for every
                            # piece type (unlike python-chess, where Move() never raises
                            # and only legal_moves membership determines legality). A
                            # careless click (e.g. two squares with no rank/file/diagonal/
                            # knight/king relationship) will hit this, so both attempts
                            # need to be guarded, not just checked against legal_moves.
                            move = None
                            try:
                                candidate = bc.Move(selected, square)
                                if candidate in legal_moves:
                                    move = candidate
                            except ValueError:
                                pass

                            if move is None:
                                try:
                                    promo = bc.Move(selected, square,promote_to=bc.QUEEN)
                                    if promo in legal_moves:
                                        move = promo
                                except ValueError:
                                    pass

                            if move is None:
                                # maybe clicked another one of your own pieces
                                piece = board[square]
                                if piece and piece.color == player_color:
                                    selected = square
                                    legal_targets = [m.destination for m in legal_moves
                                                      if m.origin == selected]
                                else:
                                    selected = None
                                    legal_targets = []

                            if move is not None:
                                board.apply(move)
                                selected = None
                                legal_targets = []
                                engine_move_if_needed()

    draw()
    clock.tick(60)

pygame.quit()