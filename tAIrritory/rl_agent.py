import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class TAIrritoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = ConvBlock(1, 64)
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(4)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 7 * 3, 7 * 3 * 7 * 3)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 7 * 3, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 7 * 3)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 7 * 3)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, max(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class TAIrritoryAgent:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = TAIrritoryNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 0
        self.gamma = 0.99
        
    def get_action(self, state, env, training=True):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            policy, value = self.model(state_tensor)
            policy = policy.exp().cpu().numpy().reshape(7, 3, 7, 3)
            
            valid_moves = []
            valid_moves_probs = []
            
            # Check all pieces for the current player
            for row in range(7):
                for col in range(3):
                    piece = state[row, col]
                    # Check if it's AI's piece (positive values)
                    if env.current_player == 2 and piece > 0:
                        possible_moves = env.possible_move_for_piece(row, col)
                        for move in possible_moves:
                            action = [row, col, move[0], move[1]]
                            valid_moves.append(action)
                            prob = float(policy[row, col, move[0], move[1]])
                            valid_moves_probs.append(prob if prob > 0 else 1e-10)
            
            # If there are valid moves, select one
            if valid_moves:
                if training and random.random() < 0.1:  # 10% random moves for exploration
                    return random.choice(valid_moves)
                else:
                    # Normalize probabilities
                    valid_moves_probs = np.array(valid_moves_probs)
                    valid_moves_probs = valid_moves_probs / valid_moves_probs.sum()
                    
                    # Choose action based on probabilities
                    try:
                        chosen_idx = np.random.choice(len(valid_moves), p=valid_moves_probs)
                        return valid_moves[chosen_idx]
                    except:
                        # Fallback to random choice if there's any numerical issue
                        return random.choice(valid_moves)
            
            return None  # Only return None if there are actually no valid moves
    
    def train(self):
        # if len(self.replay_buffer) < self.batch_size:
        #     return
        
        self.model.train()
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards = zip(*batch)
        
        state_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        action_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        reward_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
        
        policy, value = self.model(state_tensor)
        
        action_indices = (action_tensor[:, 0] * 3 + action_tensor[:, 1]) * (7 * 3) + \
                        (action_tensor[:, 2] * 3 + action_tensor[:, 3])
        policy_loss = F.nll_loss(policy, action_indices)
        value_loss = F.mse_loss(value.squeeze(), reward_tensor)
        
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path='tairritory_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path='tairritory_model.pth'):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(path, weights_only=True)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a new model.")
        if torch.cuda.is_available():
            checkpoint = torch.load(path, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])