import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from collections import deque
import random
import cv2

# ----- Hyperparameters -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPISODES = 200  # Increase for better training
MAX_STEPS = 1000    # maximum steps per episode
REPLAY_BUFFER_SIZE = 10000
TRAINING_EPOCHS = 5  # number of training epochs per update

# Loss weights
LAMBDA_DYN = 1.0
LAMBDA_REC = 1.0
LAMBDA_REG = 1e-3

# ----- Utility Functions -----
def preprocess(observation):
    """
    Convert observation (RGB image) to a normalized torch tensor.
    Resize to 84x84 for faster processing.
    """
    # Resize to 84x84
    observation = cv2.resize(observation, (84, 84))
    observation = observation.astype(np.float32) / 255.0  # scale to [0,1]
    # Rearrange axes: [H, W, C] -> [C, H, W]
    observation = np.transpose(observation, (2, 0, 1))
    return torch.tensor(observation, device=DEVICE)

# ----- Network Definitions -----
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        # A simple CNN encoder for image observations
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 84 -> 42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 42 -> 21
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 21 -> 10
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 10 -> 5
            nn.ReLU()
        )
        # Adjust fc input size to 256*5*5 = 6400
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        # Adjust fc output to 256*5*5 = 6400
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 5 -> 10
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 10 -> 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 20 -> 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 40 -> 80
            nn.Sigmoid()  # since images scaled between 0 and 1
        )
        
    def forward(self, z):
        x = self.fc(z)
        # reshape to (batch_size, 256, 5, 5)
        x = x.view(-1, 256, 5, 5)
        x_recon = self.deconv(x)
        # The output here is 80x80; resize to 84x84 to match our preprocessed image dimensions.
        x_recon = nn.functional.interpolate(x_recon, size=(84,84), mode='bilinear', align_corners=False)
        return x_recon

class InverseDynamics(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(InverseDynamics, self).__init__()
        # Takes concatenated latent representations and outputs a latent action
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, z_t, z_tp1):
        z_cat = torch.cat([z_t, z_tp1], dim=1)
        u_latent = self.fc(z_cat)
        return u_latent

class ForwardDynamics(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(ForwardDynamics, self).__init__()
        # Takes current latent state and latent action and predicts next latent state
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, z_t, u_latent):
        z_cat = torch.cat([z_t, u_latent], dim=1)
        z_tp1_pred = self.fc(z_cat)
        return z_tp1_pred

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, next_state):
        self.buffer.append((state, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, next_states = zip(*batch)
        # Stack into tensors
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        return states, next_states
    
    def __len__(self):
        return len(self.buffer)

# ----- Training Setup -----
def train():
    # Initialize networks
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    inverse_dyn = InverseDynamics().to(DEVICE)
    forward_dyn = ForwardDynamics().to(DEVICE)
    
    # Set optimizers - here we combine all parameters
    optimizer = optim.Adam(list(encoder.parameters()) +
                             list(decoder.parameters()) +
                             list(inverse_dyn.parameters()) +
                             list(forward_dyn.parameters()),
                             lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    # Create environment and replay buffer
    env = gym.make("CarRacing-v2", render_mode=None)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # Collect data from environment
    print("Collecting data...")
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = preprocess(state)
        episode_reward = 0
        done = False
        step = 0
        while not done and step < MAX_STEPS:
            # For data collection, choose a random action (we will ignore the action since we learn dynamics from images)
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_proc = preprocess(next_state)
            buffer.push(state, next_state_proc)
            state = next_state_proc
            episode_reward += reward
            done = terminated or truncated
            step += 1
        print(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {episode_reward}")

    print("Data collection complete. Starting training...")

    # Training loop: sample from replay buffer
    num_batches = len(buffer) // BATCH_SIZE
    for epoch in range(TRAINING_EPOCHS):
        epoch_loss = 0.0
        for _ in range(num_batches):
            states, next_states = buffer.sample(BATCH_SIZE)
            # Forward pass: get latent representations
            z_t = encoder(states)
            z_tp1 = encoder(next_states)
            
            # Inverse Dynamics: infer latent action from consecutive latent states
            u_latent = inverse_dyn(z_t, z_tp1)
            
            # Forward Dynamics: predict next latent state given current latent state and latent action
            z_tp1_pred = forward_dyn(z_t, u_latent)
            
            # Losses:
            # 1. Dynamics loss (latent space prediction)
            dyn_loss = mse_loss(z_tp1_pred, z_tp1)
            
            # 2. Reconstruction loss for current and next observations
            recon_t = decoder(z_t)
            recon_tp1 = decoder(z_tp1)
            rec_loss = mse_loss(recon_t, states) + mse_loss(recon_tp1, next_states)
            
            # 3. Regularization loss: simple L2 penalty on the latent codes to encourage bounded representations
            reg_loss = torch.mean(z_t**2) + torch.mean(z_tp1**2)
            
            total_loss = LAMBDA_DYN * dyn_loss + LAMBDA_REC * rec_loss + LAMBDA_REG * reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{TRAINING_EPOCHS}, Loss: {avg_loss:.6f}")
    
    print("Training complete.")
    
    # Save the models (or you can return them directly)
    torch.save(encoder.state_dict(), "checkpoints/encoder.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder.pth")
    torch.save(inverse_dyn.state_dict(), "checkpoints/inverse_dynamics.pth")
    torch.save(forward_dyn.state_dict(), "checkpoints/forward_dynamics.pth")
    
    print("Models saved.")

if __name__ == '__main__':
    train()
