import numpy as np
import gymnasium as gym

class TAIrritoryEnv(gym.Env):

    def __init__(self):
        super(TAIrritoryEnv, self).__init__()
        self.board = np.zeros((7, 3), dtype=int)
        self._initialize_board()
        self.action_space = gym.spaces.Box(low=0, high=np.array([6, 2, 6, 2]), shape=(4,), dtype=int)
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(7, 3), dtype=np.int64)
        self.current_player = 1
        self.done = False
        self.winner = None
    
    def _initialize_board(self):
        self.board[0] = [2, 1, 2]
        self.board[1] = [1, 2, 1]
        self.board[5] = [-1, -2, -1]
        self.board[6] = [-2, -1, -2]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((7, 3), dtype=int)
        self._initialize_board()
        self.current_player = 1
        self.done, self.winner = False, None
        return self.board, {}
    
    def possible_move_for_piece(self, row, col) -> list:
        piece = self.board[(row, col)]
        possible_moves = []
        
        if self.current_player == 1:
            # P1 territory: Rows 4, 5, 6
            if 4 <= row <= 6:
                if piece == -1: 
                    if col-1 >= 0 and self.board[(row - 1, col-1)] in [0, 2]:
                        possible_moves.append([row - 1, col-1])
                    
                    if self.board[(row - 1, col)] in [0, 2]:
                        possible_moves.append([row - 1, col])
                    
                    if col+1 < 3 and self.board[(row - 1, col + 1)] in [0, 2]:
                        possible_moves.append([row - 1, col + 1])
                
                elif piece == -2: 
                    if col-1 >= 0 and self.board[(row - 1, col-1)] in [0, 1]:
                        possible_moves.append([row - 1, col-1])
                    
                    if self.board[(row - 1, col)] in [0, 1]:
                        possible_moves.append([row - 1, col])
                    
                    if col+1 < 3 and self.board[(row - 1, col+1)] in [0, 1]:
                        possible_moves.append([row - 1, col+1])
        
        elif self.current_player == 2:
            # P2 territory: Rows 0, 1, 2
            if 0 <= row <= 2:
                if piece == 1:
                    if col-1 >= 0 and self.board[(row + 1, col-1)] in [-2, 0]:
                        possible_moves.append([row + 1, col-1])
                    
                    if self.board[(row + 1, col)] in [-2, 0]:
                        possible_moves.append([row + 1, col])
                    
                    if col+1 < 3 and self.board[(row + 1, col+1)] in [-2, 0]:
                        possible_moves.append([row + 1, col+1])
                
                elif piece == 2:
                    if col-1 >= 0 and self.board[(row + 1, col-1)] in [-1, 0]:
                        possible_moves.append([row + 1, col-1])
                    
                    if self.board[(row + 1, col)] in [-1, 0]:
                        possible_moves.append([row + 1, col])
                    
                    if col+1 < 3 and self.board[(row + 1, col+1)] in [-1, 0]:
                        possible_moves.append([row + 1, col+1])
        
        return possible_moves
    
    def _is_game_over(self, p_type):
        for row in range(7):
            for col in range(3):
                if (p_type == 1 and (self.board[row, col] == -1 or self.board[row, col] == -2)) or \
                (p_type == 2 and (self.board[row, col] == 1 or self.board[row, col] == 2)):
                    if len(self.possible_move_for_piece(row, col)) > 0:
                        return False
        return True

    def _calculate_reward(self):
        p1_count = np.sum((self.board[3] == -1) | (self.board[3] == -2))
        p2_count = np.sum((self.board[3] == 1) | (self.board[3] == 2))
        return 0 if p1_count == p2_count else 1 if p1_count < p2_count else -1
    
    def step(self, action):
        self.board[(action[0], action[1])], self.board[(action[2], action[3])] = \
        self.board[(action[2], action[3])], self.board[(action[0], action[1])]
        self.current_player = 1 if self.current_player == 2 else 2
        terminated = self._is_game_over(self.current_player)
        reward = self._calculate_reward() if terminated else 0
        truncated = False
        info = {}
        return self.board, reward, terminated, truncated, info
    
    def render(self, mode=None):
        if mode == "human":
            print("\n".join(" ".join(str(cell) for cell in row) for row in self.board))

    def close(self):
        pass

    
