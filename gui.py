import pygame
import chess
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
WHITE = (245, 245, 220)
GRAY = (112, 128, 144)

def draw_squares(win):
    win.fill(WHITE)
    for row in range(ROWS):
        # Alternate starting color for each row
        for col in range(row % 2, COLS, 2):
            pygame.draw.rect(win, GRAY, (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Pygame Chess Board')
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        draw_squares(win)
        pygame.display.update()

    pygame.quit()

main()