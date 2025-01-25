import pygame
import math
import gymnasium as gym
from gymnasium.envs.registration import register
import time

register(
    id="gymnasium_env/tAIrritory-v0",
    entry_point="Game_Env:TAIrritoryEnv",
)

class TAIrritoryPygame:

    def __init__(self, board_size=7, board_width=350, board_height=700):
        pygame.init()
        self.board_size = board_size
        self.board_width = board_width
        self.board_height = board_height
        self.tile_width = board_width // 3
        self.tile_height = board_height // board_size
        # Adjust screen height to make space for the scoreboard
        self.screen = pygame.display.set_mode((board_width, board_height + 50))
        pygame.display.set_caption("tAIrritory")
        self.clock = pygame.time.Clock()
        # Load game environment
        self.env = gym.make("gymnasium_env/tAIrritory-v0")
        self.obs, _ = self.env.reset()
        # Load images
        self.p2_01 = pygame.transform.scale(
            pygame.image.load("images/02.svg"), (self.tile_width, self.tile_height)
        )
        self.p2_02 = pygame.transform.scale(
            pygame.image.load("images/03.svg"), (self.tile_width, self.tile_height)
        )
        self.p1_01 = pygame.transform.scale(
            pygame.image.load("images/00.svg"), (self.tile_width, self.tile_height)
        )
        self.p1_02 = pygame.transform.scale(
            pygame.image.load("images/01.svg"), (self.tile_width, self.tile_height)
        )
        # Game state
        self.selected_piece = None
        self.ai_wins = 0
        self.human_wins = 0

    def get_piece_image(self, piece_value):
        piece_map = {
            1: self.p2_01,
            2: self.p2_02,
            -1: self.p1_01,
            -2: self.p1_02,
        }
        return piece_map.get(piece_value)

    def draw_board(self):
        self.screen.fill((220, 220, 220))  # Background color
        # Draw the tiles
        for row in range(7):
            for col in range(3):
                rect = pygame.Rect(
                    col * self.tile_width,
                    row * self.tile_height,
                    self.tile_width,
                    self.tile_height,
                )
                color = (228, 228, 222) if (row + col) % 2 == 0 else (11, 11, 11)
                pygame.draw.rect(self.screen, color, rect)
                # Draw pieces
                piece_value = self.obs[row, col]
                piece_image = self.get_piece_image(piece_value)
                if piece_image:
                    piece_rect = piece_image.get_rect(
                        center=(
                            col * self.tile_width + self.tile_width // 2,
                            row * self.tile_height + self.tile_height // 2,
                        )
                    )
                    self.screen.blit(piece_image, piece_rect)
        # Add transparent blue shade for rows 0, 1, 2
        blue_surface = pygame.Surface((self.board_width, self.tile_height * 3), pygame.SRCALPHA)
        blue_surface.fill((0, 0, 230, 50))  # Blue with transparency
        self.screen.blit(blue_surface, (0, 0))
        # Add transparent red shade for rows 4, 5, 6
        red_surface = pygame.Surface((self.board_width, self.tile_height * 3), pygame.SRCALPHA)
        red_surface.fill((230, 0, 0, 50))  # Red with transparency
        self.screen.blit(red_surface, (0, self.tile_height * 4))
        # Highlight possible moves for selected piece
        if self.selected_piece:
            row, col = self.selected_piece
            possible_moves = self.env.unwrapped.possible_move_for_piece(row, col)
            # Glowing effect
            time_ms = pygame.time.get_ticks()
            glow_alpha = int(100 + 50 * math.sin(time_ms * 0.01))
            for move in possible_moves:
                move_rect = pygame.Rect(
                    move[1] * self.tile_width,
                    move[0] * self.tile_height,
                    self.tile_width,
                    self.tile_height,
                )
                # Glowing green fill
                glow_surface = pygame.Surface((self.tile_width, self.tile_height), pygame.SRCALPHA)
                glow_surface.fill((100, 255, 100, glow_alpha))
                self.screen.blit(glow_surface, move_rect)
                # Green border
                pygame.draw.rect(self.screen, (50, 200, 50), move_rect, 4)
        # Draw the score
        self.draw_score()
        pygame.display.flip()

    def draw_score(self):
        # Fonts and Colors
        font = pygame.font.SysFont("Sixtyfour Convergence", 22, bold=True)
        bg_color = (245, 245, 245)  # Light background
        human_text_color = (211, 47, 47)  # Red for humans
        ai_text_color = (25, 118, 210)  # Blue for AI
        border_color = (200, 200, 200)  # Subtle border color
        # Background for score area
        score_area_rect = pygame.Rect(0, self.board_height, self.board_width, 60)
        pygame.draw.rect(self.screen, bg_color, score_area_rect)
        # Border for the score area
        pygame.draw.line(self.screen, border_color, (0, self.board_height), (self.board_width, self.board_height), 2)
        # Icons (optional: you can replace with actual images for more styling)
        human_icon_rect = pygame.Rect(10, self.board_height + 15, 30, 30)
        ai_icon_rect = pygame.Rect(self.board_width - 40, self.board_height + 15, 30, 30)
        pygame.draw.ellipse(self.screen, human_text_color, human_icon_rect)  # Circle for humans
        pygame.draw.ellipse(self.screen, ai_text_color, ai_icon_rect)  # Circle for AI
        # Render text for Human and AI
        human_wins_text = font.render(f"Humans: {self.human_wins}", True, human_text_color)
        ai_wins_text = font.render(f"{self.ai_wins} :AI", True, ai_text_color)
        # Positioning Text
        self.screen.blit(human_wins_text, (50, self.board_height + 20))  # Left-aligned
        ai_text_width = ai_wins_text.get_width()
        ai_text_x = self.board_width - ai_text_width - 50  # Right-aligned
        self.screen.blit(ai_wins_text, (ai_text_x, self.board_height + 20))

    def handle_click(self, pos):
        col = pos[0] // self.tile_width
        row = pos[1] // self.tile_height
        piece_value = self.obs[row, col]
        if self.selected_piece is None:
            # Select a piece if it belongs to the current player
            if (self.env.unwrapped.current_player == 1 and piece_value < 0) or \
               (self.env.unwrapped.current_player == 2 and piece_value > 0):
                self.selected_piece = (row, col)
        # Automatically switch piece selected
        elif (self.env.unwrapped.current_player == 1 and piece_value < 0) or \
               (self.env.unwrapped.current_player == 2 and piece_value > 0) and \
               self.selected_piece != (row, col):
            self.selected_piece = (row, col)
        else:
            # Try to move the selected piece
            action = [self.selected_piece[0], self.selected_piece[1], row, col]
            try:
                possible_moves = self.env.unwrapped.possible_move_for_piece(self.selected_piece[0], self.selected_piece[1])
                if [row, col] in possible_moves:
                    self.obs, reward, terminated, _, _ = self.env.step(action)
                    if terminated:
                        if reward == 1:  # AI win
                            self.ai_wins += 1
                        elif reward == -1:  # Human win
                            self.human_wins += 1
                        self.reset_game()
                        return
                else:
                    print("Invalid move")
            except Exception as e:
                print(f"Error: {e}")
            self.selected_piece = None

    def reset_game(self):
        self.obs, _ = self.env.reset()
        self.selected_piece = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            self.draw_board()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    game = TAIrritoryPygame()
    game.run()
