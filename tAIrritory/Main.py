import pygame
import math
import gymnasium as gym
from gymnasium.envs.registration import register
from rl_agent import TAIrritoryAgent  

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
        self.tile_height = (board_height - 50) // board_size  # Adjust for score area
        # Initialize the RL agent
        self.agent = TAIrritoryAgent()
        try:
            self.agent.load()  # Load pretrained model if available
        except:
            print("No pretrained model found. Starting with a new model.")
        self.screen = pygame.display.set_mode((board_width, board_height + 50))
        pygame.display.set_caption("tAIrritory")
        self.clock = pygame.time.Clock()
        # Load game environment
        self.env = gym.make("gymnasium_env/tAIrritory-v0")
        self.obs, _ = self.env.reset()
        # Load images
        self.p2_01 = pygame.transform.smoothscale(
            pygame.image.load("images/p2_01.svg"), (self.tile_width, self.tile_height)
        )
        self.p2_02 = pygame.transform.smoothscale(
            pygame.image.load("images/p2_02.svg"), (self.tile_width, self.tile_height)
        )
        self.p1_01 = pygame.transform.smoothscale(
            pygame.image.load("images/p1_01.svg"), (self.tile_width, self.tile_height)
        )
        self.p1_02 = pygame.transform.smoothscale(
            pygame.image.load("images/p1_02.svg"), (self.tile_width, self.tile_height)
        )
        # Game state
        self.selected_piece = None
        self.ai_wins = 0
        self.human_wins = 0
        self.game_over = False
        self.winner = None

        # Colors and fonts for UI
        self.colors = {
            'background': (245, 245, 245),
            'board': (228, 228, 222),
            'highlights': (100, 230, 100, 120),
            'inactive_piece': (200, 200, 200),
            'scores': {
                'background': (230, 230, 230),
                'text': (50, 50, 50)
            },
            'borders': (200, 200, 200)
        }
        # self.font = pygame.font.Font("fonts/Sixtyfour Convergence.ttf", 24)

    def get_piece_image(self, piece_value):
        piece_map = {
            1: self.p2_01,
            2: self.p2_02,
            -1: self.p1_01,
            -2: self.p1_02,
        }
        return piece_map.get(piece_value)

    def draw_score(self):
        font = pygame.font.SysFont(None, 24)
        human_text = font.render(f"Human: {self.human_wins}", True, (0, 0, 0))
        ai_text = font.render(f"AI: {self.ai_wins}", True, (0, 0, 0))
        turn_text = font.render(f"Turn: {'Human' if self.env.unwrapped.current_player == 1 else 'AI'}", True, (0, 0, 0))
        self.screen.blit(human_text, (10, self.board_height + 10))
        self.screen.blit(ai_text, (self.board_width - ai_text.get_width() - 10, self.board_height + 10))
        self.screen.blit(turn_text, (self.board_width // 2 - turn_text.get_width() // 2, self.board_height + 10))

    def show_how_to_play(self):
        font = pygame.font.SysFont(None, 28)
        text_lines = [
            "tAIrritory Rules:",
            "1. Move pieces forward in your territory.",
            "2. Interchange places with opposite pieces.",
            "3. Block movement with same-type pieces.",
            "4. Game ends when no moves are possible.",
            "5. Player with most pieces in R4 wins."
        ]
        dialog_surface = pygame.Surface((self.board_width - 20, 200), pygame.SRCALPHA)
        dialog_surface.fill((200, 200, 200, 220))
        y_offset = 10
        for line in text_lines:
            rendered_text = font.render(line, True, (0, 0, 0))
            dialog_surface.blit(rendered_text, (10, y_offset))
            y_offset += 30
        self.screen.blit(dialog_surface, (10, self.board_height // 2 - 100))
        pygame.display.flip()
        pygame.time.wait(5000)

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

    def handle_click(self, pos):
        if self.game_over:
            return
        if self.env.unwrapped.current_player == 2:
            # AI's turn
            action = self.agent.get_action(self.obs, self.env.unwrapped, training=True)
            if action is not None:
                old_state = self.obs.copy()
                self.obs, reward, terminated, _, _ = self.env.step(action)
                self.agent.replay_buffer.push(old_state, action, reward)
                if terminated:  # Game is over because human (next player) has no moves
                    self.end_game(reward)
            return
        else:
            # Human's turn
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
                        old_state = self.obs.copy()
                        self.obs, reward, terminated, _, _ = self.env.step(action)
                        if terminated:  # Game is over because AI (next player) has no moves
                            self.end_game(reward)
                    else:
                        print("Invalid move")
                except Exception as e:
                    print(f"Error: {e}")
                self.selected_piece = None
                
    def end_game(self, reward):
        self.game_over = True
        if reward == 1:
            self.winner = "AI"
            self.ai_wins += 1
        elif reward == -1:
            self.winner = "Human"
            self.human_wins += 1
        else:
            self.winner = "Draw"
        self.agent.train()
        self.agent.save()
        self.show_game_over_screen()

    def show_game_over_screen(self):
        font = pygame.font.SysFont(None, 48)
        text = font.render(f"Game Over! Winner: {self.winner}", True, (0, 0, 0))
        self.screen.fill((245, 245, 245))
        self.screen.blit(text, (self.board_width // 2 - text.get_width() // 2, self.board_height // 2 - text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(3000)
        self.reset_game()

    def reset_game(self):
        self.obs, _ = self.env.reset()
        self.selected_piece = None
        self.game_over = False
        self.winner = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if 10 <= event.pos[0] <= 110 and self.board_height + 50 <= event.pos[1] <= self.board_height + 90:
                        self.show_how_to_play()
                    else:
                        self.handle_click(event.pos)
            self.draw_board()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    game = TAIrritoryPygame()
    game.run()
