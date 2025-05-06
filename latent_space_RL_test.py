import gymnasium as gym
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Make sure to use the same DEVICE and hyperparameters as before.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64

# Define the same network architectures as used in training.
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 5, 5)
        x_recon = self.deconv(x)
        x_recon = nn.functional.interpolate(x_recon, size=(84,84), mode='bilinear', align_corners=False)
        return x_recon

class InverseDynamics(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(InverseDynamics, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, z_t, u_latent):
        z_cat = torch.cat([z_t, u_latent], dim=1)
        z_tp1_pred = self.fc(z_cat)
        return z_tp1_pred

def preprocess(observation):
    """
    Resize to 84x84, normalize, and arrange as torch tensor.
    """
    observation = cv2.resize(observation, (84, 84))
    observation = observation.astype(np.float32) / 255.0
    observation = np.transpose(observation, (2, 0, 1))
    return torch.tensor(observation, device=DEVICE)

# Load the models
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
inverse_dyn = InverseDynamics().to(DEVICE)
forward_dyn = ForwardDynamics().to(DEVICE)

encoder.load_state_dict(torch.load("checkpoints/encoder.pth", map_location=DEVICE))
decoder.load_state_dict(torch.load("checkpoints/decoder.pth", map_location=DEVICE))
inverse_dyn.load_state_dict(torch.load("checkpoints/inverse_dynamics.pth", map_location=DEVICE))
forward_dyn.load_state_dict(torch.load("checkpoints/forward_dynamics.pth", map_location=DEVICE))

# Set all models to evaluation mode
encoder.eval()
decoder.eval()
inverse_dyn.eval()
forward_dyn.eval()

# Create a gym environment for testing
env = gym.make("CarRacing-v2", render_mode='rgb_array')

# Test one episode
state, _ = env.reset()
state_proc = preprocess(state).unsqueeze(0)  # add batch dim

# For visual testing, we will do both reconstruction and latent dynamics prediction.
with torch.no_grad():
    # 1. Reconstruction test
    z_t = encoder(state_proc)
    recon_state = decoder(z_t)
    recon_state_np = recon_state.squeeze(0).cpu().numpy().transpose(1,2,0)
    
    # Plot original vs reconstructed
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(recon_state_np)
    plt.axis('off')
    plt.show()
    
    # 2. Latent dynamics test:
    # Take one step using a random action and compare predicted latent state vs actual.
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state_proc = preprocess(next_state).unsqueeze(0)
    
    z_tp1 = encoder(next_state_proc)
    # Compute latent action from consecutive latent states
    u_latent = inverse_dyn(z_t, z_tp1)
    # Predict next latent state from current latent state and latent action
    z_tp1_pred = forward_dyn(z_t, u_latent)
    # Decode the predicted latent state back to observation space
    recon_next_state = decoder(z_tp1_pred)
    
    # Convert for visualization
    next_state_np = cv2.cvtColor(next_state, cv2.COLOR_BGR2RGB)
    recon_next_state_np = recon_next_state.squeeze(0).cpu().numpy().transpose(1,2,0)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Current State")
    plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title("Actual Next State")
    plt.imshow(next_state_np)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.title("Predicted Next State")
    plt.imshow(recon_next_state_np)
    plt.axis('off')
    plt.show()

env.close()
