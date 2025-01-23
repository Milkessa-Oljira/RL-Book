import pygame
import gymnasium as gym
import numpy as np

class TAIrritoryPygame:
    def __init__(self, board_size=7, board_width=400, board_height=700):
        pygame.init()
        self.board_size = board_size
        self.board_width = board_width
        self.board_height = board_height
        self.screen = pygame.display.set_mode((board_width, board_height))
        pygame.display.set_caption("TAIrritory")
        
        # Import your TAIrritoryEnv
        from Game_Env import TAIrritoryEnv
        self.env = TAIrritoryEnv()
        self.obs, _ = self.env.reset()
        
        self.tile_width = board_width // 3
        self.tile_height = board_height // 7
        
        # Load images
        self.black_pawn = pygame.transform.scale(
            pygame.image.load('images/black-pawn.png'), 
            (self.tile_width, self.tile_height)
        )
        self.black_king = pygame.transform.scale(
            pygame.image.load('images/black-king.png'), 
            (self.tile_width, self.tile_height)
        )
        self.red_pawn = pygame.transform.scale(
            pygame.image.load('images/red-pawn.png'), 
            (self.tile_width, self.tile_height)
        )
        self.red_king = pygame.transform.scale(
            pygame.image.load('images/red-king.png'), 
            (self.tile_width, self.tile_height)
        )
        
        self.clock = pygame.time.Clock()
        self.selected_piece = None
        
    def get_piece_image(self, piece_value):
        piece_map = {
            1: self.black_pawn,
            2: self.black_king,
            -1: self.red_pawn,
            -2: self.red_king
        }
        return piece_map.get(piece_value)
    
    def draw_board(self):
        self.screen.fill((220, 220, 220))
        for row in range(7):
            for col in range(3):
                rect = pygame.Rect(
                    col * self.tile_width, 
                    row * self.tile_height, 
                    self.tile_width, 
                    self.tile_height
                )
                color = (255, 255, 255) if (row + col) % 2 == 0 else (100, 100, 100)
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw pieces
                piece_value = self.obs[row, col]
                piece_image = self.get_piece_image(piece_value)
                if piece_image:
                    piece_rect = piece_image.get_rect(
                        center=(
                            col * self.tile_width + self.tile_width // 2,
                            row * self.tile_height + self.tile_height // 2
                        )
                    )
                    self.screen.blit(piece_image, piece_rect)
        
        pygame.display.flip()
    
    def handle_click(self, pos):
        col = pos[0] // self.tile_width
        row = pos[1] // self.tile_height
        
        if self.selected_piece is None:
            # Select a piece if it belongs to current player
            piece_value = self.obs[row, col]
            if (self.env.current_player == 1 and piece_value > 0) or \
               (self.env.current_player == 2 and piece_value < 0):
                self.selected_piece = (row, col)
        else:
            # Try to move the selected piece
            action = [self.selected_piece[0], self.selected_piece[1], row, col]
            try:
                possible_moves = self.env.possible_move_for_piece(self.selected_piece[0], self.selected_piece[1])
                if [row, col] in possible_moves:
                    self.obs, reward, terminated, truncated, _ = self.env.step(action)
                    if terminated:
                        print(f"Game over! Reward: {reward}")
                        return False
                else:
                    print("Invalid move")
            except Exception as e:
                print(f"Error: {e}")
            
            self.selected_piece = None
        
        return True
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    running = self.handle_click(event.pos)
            
            self.draw_board()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = TAIrritoryPygame()
    game.run()