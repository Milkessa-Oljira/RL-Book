"""
Latent Action Framework for Reinforcement Learning
Applied to Gymnasium CarRacing Environment

This implementation follows the architecture described in:
"A Latent-Action Framework for Reinforcement Learning Combining Generalization and Precise Localization"

Components implemented:
1. State Embedder - Transforms raw observations into latent representations
2. Latent Action Generator - Learns abstract actions in latent space
3. Deterministic Decoder - Maps latent actions to concrete actions
4. Model-Predictive Control - Plans in latent space
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
from typing import Tuple, List, Dict, Optional, Union, Any

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# For GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    # Environment
    env_name = "CarRacing-v2"
    input_shape = (3, 96, 96)  # RGB images from CarRacing
    action_dim = 3  # (steering, gas, brake)
    
    # Architecture
    latent_dim = 64
    hidden_dim = 256
    
    # Training
    batch_size = 64
    buffer_size = 100000
    min_buffer_size = 5000
    learning_rate = 3e-4
    gamma = 0.99
    tau = 0.001  # Target network update rate
    
    # MPC Parameters
    horizon = 10
    n_samples = 128
    top_k = 16
    
    # Training schedule
    total_steps = 500000
    exploration_steps = 50000
    
    # Checkpoints
    checkpoint_dir = "./checkpoints/"
    checkpoint_interval = 10000
    
    # Evaluation
    eval_interval = 5000
    eval_episodes = 3
    render = False


# =============================================================================
# 1. STATE EMBEDDER
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class StateEmbedder(nn.Module):
    """
    Transforms raw observations into latent representations using
    energy-based or contrastive methods.
    """
    def __init__(self, input_shape, latent_dim):
        super(StateEmbedder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Convolutional encoder for image observations
        self.encoder = nn.Sequential(
            ConvBlock(input_shape[0], 32, 4, 2, 1),  # 96x96 -> 48x48
            ConvBlock(32, 64, 4, 2, 1),             # 48x48 -> 24x24
            ConvBlock(64, 128, 4, 2, 1),            # 24x24 -> 12x12
            ConvBlock(128, 256, 4, 2, 1),           # 12x12 -> 6x6
            nn.Flatten(),                           # 256 * 6 * 6
            nn.Linear(256 * 6 * 6, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )
        
        # Projection for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )
        
    def forward(self, x):
        """
        Forward pass through the embedder
        Args:
            x: Raw state observation tensor [B, C, H, W]
        Returns:
            latent: Latent representation [B, latent_dim]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(device)
            
        # Normalize if image data
        if x.dim() == 4 and x.shape[1] == 3:
            x = x / 255.0
            
        latent = self.encoder(x)
        return latent
    
    def get_projection(self, x):
        """Get projection for contrastive learning"""
        latent = self.forward(x)
        projection = self.projector(latent)
        return F.normalize(projection, dim=1)
    
    def contrastive_loss(self, z1, z2, temperature=0.1):
        """
        Calculates the InfoNCE contrastive loss
        Args:
            z1, z2: Batch of projected representations [B, projection_dim]
            temperature: Temperature parameter
        """
        batch_size = z1.shape[0]
        
        # Feature representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Gather all representations across devices if using distributed training
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                              representations.unsqueeze(0), 
                                              dim=2)
        
        # Mask for positive pairs
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Remove diagonal (self-similarity)
        mask = (~torch.eye(batch_size * 2, dtype=bool, device=device))
        nominator = torch.exp(positives / temperature)
        denominator = mask * torch.exp(similarity_matrix / temperature)
        
        # InfoNCE loss
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.mean(all_losses)
        return loss


# =============================================================================
# 2. LATENT ACTION GENERATOR
# =============================================================================

class LatentActionGenerator(nn.Module):
    """
    Learns an abstract action as a function of successive latent states.
    """
    def __init__(self, latent_dim, hidden_dim):
        super(LatentActionGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Neural network to predict the latent action
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x_t, x_t1):
        """
        Predict the latent action that resulted in transition from x_t to x_t1
        Args:
            x_t: Current latent state [B, latent_dim]
            x_t1: Next latent state [B, latent_dim]
        Returns:
            latent_action: Predicted latent action [B, latent_dim]
        """
        # Concatenate latent states
        x_combined = torch.cat([x_t, x_t1], dim=1)
        latent_action = self.network(x_combined)
        return latent_action
    
    def loss(self, x_t, x_t1, latent_action=None):
        """
        Compute the loss for the latent action generator
        Args:
            x_t: Current latent state [B, latent_dim]
            x_t1: Next latent state [B, latent_dim]
            latent_action: Optional pre-computed latent action
        Returns:
            loss: MSE loss between predicted next state and actual next state
        """
        if latent_action is None:
            latent_action = self.forward(x_t, x_t1)
        
        # Predict next state by adding latent action to current state
        predicted_x_t1 = x_t + latent_action
        
        # Compute mean squared error
        loss = F.mse_loss(predicted_x_t1, x_t1)
        return loss


# =============================================================================
# 3. DETERMINISTIC DECODER
# =============================================================================

class DeterministicDecoder(nn.Module):
    """
    Maps latent actions to concrete actions in the environment.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(DeterministicDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Neural network for decoding
        self.network = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim),  # latent_state + latent_action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Action bounds for CarRacing
        self.action_bounds = {
            'steering': (-1.0, 1.0),  # Left, Right
            'gas': (0.0, 1.0),        # Gas
            'brake': (0.0, 1.0)       # Brake
        }
        
    def forward(self, latent_state, latent_action):
        """
        Decode latent action to concrete action
        Args:
            latent_state: Current latent state [B, latent_dim]
            latent_action: Latent action [B, latent_dim]
        Returns:
            action: Concrete action [B, action_dim]
        """
        # Concatenate latent state and latent action
        x_combined = torch.cat([latent_state, latent_action], dim=1)
        
        # Decode to raw action
        raw_action = self.network(x_combined)
        
        # Apply action bounds
        steering = torch.tanh(raw_action[:, 0])  # -1 to 1
        gas = torch.sigmoid(raw_action[:, 1])    # 0 to 1
        brake = torch.sigmoid(raw_action[:, 2])  # 0 to 1
        
        # Combine into final action
        action = torch.stack([steering, gas, brake], dim=1)
        return action
    
    def loss(self, latent_state, latent_action, target_next_latent, embedder):
        """
        Compute loss for the decoder by evaluating how well the decoded action
        transitions from current to next latent state
        """
        # Get concrete action
        actions = self.forward(latent_state, latent_action)
        
        # Check if actions are within bounds and apply penalty if not
        penalty = 0.0
        actions_np = actions.detach().cpu().numpy()
        for i, bounds in enumerate(self.action_bounds.values()):
            low, high = bounds
            out_of_bounds = np.logical_or(actions_np[:, i] < low, actions_np[:, i] > high)
            penalty += out_of_bounds.sum()
        
        # The penalty is applied to the loss function
        penalty_factor = 0.1 * penalty / latent_state.shape[0]
        
        # Main loss: the latent dynamics should predict the next latent state
        predicted_next_latent = latent_state + latent_action
        loss = F.mse_loss(predicted_next_latent, target_next_latent) + penalty_factor
        
        return loss


# =============================================================================
# 4. MODEL-PREDICTIVE CONTROL (MPC)
# =============================================================================

class LatentMPC:
    """
    Model-Predictive Control in the latent space.
    """
    def __init__(self, latent_dim, horizon, n_samples, top_k):
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.top_k = top_k
        
    def optimize(self, current_latent, target_latent, latent_action_generator, decoder, state_embedder):
        """
        Optimize the sequence of latent actions to reach the target latent state
        using a shooting method.
        
        Args:
            current_latent: Current latent state [latent_dim]
            target_latent: Target latent state [latent_dim]
            latent_action_generator: Model for generating latent actions
            decoder: Model for decoding latent actions to concrete actions
            state_embedder: Model for embedding states
            
        Returns:
            optimal_action: Optimal concrete action for the current step
        """
        # Expand current latent for batch processing
        batch_current = current_latent.repeat(self.n_samples, 1)
        
        # Initialize trajectory of latent states and costs
        trajectory_latent_states = [batch_current]
        trajectory_latent_actions = []
        trajectory_costs = torch.zeros(self.n_samples, device=device)
        
        # Sample latent actions for the horizon
        for h in range(self.horizon):
            # Sample random actions in latent space
            latent_actions = torch.randn(self.n_samples, self.latent_dim, device=device) * 0.5
            
            # Predict next latent state
            next_latent = trajectory_latent_states[-1] + latent_actions
            
            # Compute costs (distance to target and action regularization)
            state_cost = F.mse_loss(next_latent, target_latent.repeat(self.n_samples, 1), reduction='none').sum(dim=1)
            action_cost = 0.1 * torch.norm(latent_actions, dim=1)
            step_cost = state_cost + action_cost
            
            # Accumulate costs
            trajectory_costs += step_cost
            
            # Store for next iteration
            trajectory_latent_states.append(next_latent)
            trajectory_latent_actions.append(latent_actions)
        
        # Select top-k trajectories
        _, top_indices = torch.topk(trajectory_costs, self.top_k, largest=False)
        
        # Refine the best actions using cross-entropy method (CEM)
        for _ in range(3):  # CEM iterations
            # Select top latent actions for the first step
            top_latent_actions = trajectory_latent_actions[0][top_indices]
            
            # Compute mean and std of top actions
            mean = torch.mean(top_latent_actions, dim=0)
            std = torch.std(top_latent_actions, dim=0) + 1e-6  # Add epsilon for stability
            
            # Sample new actions around the mean
            latent_actions = torch.normal(mean.repeat(self.n_samples, 1), std.repeat(self.n_samples, 1))
            
            # Reset and recompute trajectories with new sampled actions
            trajectory_latent_states = [batch_current]
            trajectory_latent_actions = [latent_actions]
            trajectory_costs = torch.zeros(self.n_samples, device=device)
            
            # Simulate rest of the horizon with new first action
            next_latent = batch_current + latent_actions
            trajectory_latent_states.append(next_latent)
            
            state_cost = F.mse_loss(next_latent, target_latent.repeat(self.n_samples, 1), reduction='none').sum(dim=1)
            action_cost = 0.1 * torch.norm(latent_actions, dim=1)
            trajectory_costs += state_cost + action_cost
            
            # Continue with random sampling for remaining steps
            for h in range(1, self.horizon):
                rand_actions = torch.randn(self.n_samples, self.latent_dim, device=device) * 0.5
                next_latent = trajectory_latent_states[-1] + rand_actions
                
                state_cost = F.mse_loss(next_latent, target_latent.repeat(self.n_samples, 1), reduction='none').sum(dim=1)
                action_cost = 0.1 * torch.norm(rand_actions, dim=1)
                step_cost = state_cost + action_cost
                
                trajectory_costs += step_cost
                trajectory_latent_states.append(next_latent)
                trajectory_latent_actions.append(rand_actions)
            
            # Update top indices
            _, top_indices = torch.topk(trajectory_costs, self.top_k, largest=False)
        
        # Get the best latent action (from the first step)
        best_idx = top_indices[0]
        best_latent_action = trajectory_latent_actions[0][best_idx].unsqueeze(0)
        
        # Decode to concrete action
        with torch.no_grad():
            concrete_action = decoder(current_latent.unsqueeze(0), best_latent_action)
            
        return concrete_action.squeeze(0), best_latent_action.squeeze(0)


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer to store transitions."""
    
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        
        # Separate out elements
        states = np.stack([exp[0] for exp in batch])
        actions = np.stack([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.stack([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


# =============================================================================
# DATA AUGMENTATION FOR CONTRASTIVE LEARNING
# =============================================================================

class ImageAugmentation:
    """Image augmentation for contrastive learning."""
    
    @staticmethod
    def random_crop(image, output_size=(96, 96)):
        """Random crop the image."""
        h, w = image.shape[1:3]
        new_h, new_w = output_size
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        image = image[:, top:top+new_h, left:left+new_w]
        return image
    
    @staticmethod
    def color_jitter(image, strength=0.1):
        """Apply color jittering."""
        # Convert to float if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        # Apply color jitter
        brightness = 1.0 + np.random.uniform(-strength, strength)
        contrast = 1.0 + np.random.uniform(-strength, strength)
        saturation = 1.0 + np.random.uniform(-strength, strength)
        
        # Apply transforms
        image = image * brightness
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        image = (image - mean) * contrast + mean
        
        # Ensure values are in valid range
        image = np.clip(image, 0.0, 1.0)
        
        # Convert back to original format if needed
        if image.dtype != np.float32:
            image = (image * 255).astype(np.uint8)
            
        return image
    
    @staticmethod
    def augment(image):
        """Apply a series of augmentations."""
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        # Apply augmentations
        if np.random.rand() < 0.8:
            image = ImageAugmentation.random_crop(image)
        if np.random.rand() < 0.8:
            image = ImageAugmentation.color_jitter(image)
            
        return image


# =============================================================================
# TRAINING PROCEDURE
# =============================================================================

class LatentActionAgent:
    """Main agent implementing the Latent Action Framework."""
    
    def __init__(self, config: Config):
        self.config = config
        self.env = None
        
        # Create models
        self.state_embedder = StateEmbedder(config.input_shape, config.latent_dim).to(device)
        self.latent_action_generator = LatentActionGenerator(config.latent_dim, config.hidden_dim).to(device)
        self.decoder = DeterministicDecoder(config.latent_dim, config.action_dim, config.hidden_dim).to(device)
        
        # Create MPC planner
        self.mpc = LatentMPC(config.latent_dim, config.horizon, config.n_samples, config.top_k)
        
        # Create optimizers
        self.embedder_optimizer = optim.Adam(self.state_embedder.parameters(), lr=config.learning_rate)
        self.lag_optimizer = optim.Adam(self.latent_action_generator.parameters(), lr=config.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.batch_size)
        
        # Create checkpoint directory
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
            
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'embedder_losses': [],
            'lag_losses': [],
            'decoder_losses': [],
            'eval_rewards': []
        }
        
    def create_env(self):
        """Create and configure the gym environment."""
        env = gym.make(self.config.env_name, continuous=True, render_mode="rgb_array")
        return env
    
    def get_action(self, state, target_latent=None, explore=False):
        """
        Get an action from the policy.
        Args:
            state: Current state observation
            target_latent: Optional target latent state
            explore: Whether to add exploration noise
        """
        # Convert state to tensor and get latent representation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            current_latent = self.state_embedder(state_tensor)
        
        # If target latent not provided, use a heuristic target
        if target_latent is None:
            # For CarRacing, we can use a simple heuristic: keep the car on the track
            # This could be improved with a separate target network or reward model
            target_latent = current_latent.clone()
            # Modify to encourage forward motion on the track
            target_latent[0, 0] += 0.5  # Arbitrary direction in latent space
        
        # Get optimal action using MPC
        action, latent_action = self.mpc.optimize(
            current_latent, target_latent, 
            self.latent_action_generator, self.decoder, self.state_embedder
        )
        
        # Add exploration noise during training
        if explore:
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            # Clip to valid range
            action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy(), latent_action.cpu().numpy()
    
    def train_embedder(self, states):
        """Train the state embedder using contrastive learning."""
        # Create augmented views
        states_view1 = []
        states_view2 = []
        
        for state in states:
            # Apply different augmentations to create two views
            view1 = torch.FloatTensor(ImageAugmentation.augment(state)).to(device)
            view2 = torch.FloatTensor(ImageAugmentation.augment(state)).to(device)
            
            states_view1.append(view1)
            states_view2.append(view2)
            
        states_view1 = torch.stack(states_view1)
        states_view2 = torch.stack(states_view2)
        
        # Get projections
        z1 = self.state_embedder.get_projection(states_view1)
        z2 = self.state_embedder.get_projection(states_view2)
        
        # Compute contrastive loss
        loss = self.state_embedder.contrastive_loss(z1, z2)
        
        # Backpropagation
        self.embedder_optimizer.zero_grad()
        loss.backward()
        self.embedder_optimizer.step()
        
        return loss.item()
    
    def train_latent_action_generator(self, states, next_states):
        """Train the latent action generator."""
        # Get latent representations
        with torch.no_grad():
            latent_states = self.state_embedder(states)
            latent_next_states = self.state_embedder(next_states)
        
        # Compute loss
        loss = self.latent_action_generator.loss(latent_states, latent_next_states)
        
        # Backpropagation
        self.lag_optimizer.zero_grad()
        loss.backward()
        self.lag_optimizer.step()
        
        return loss.item()
    
    def train_decoder(self, states, actions, next_states):
        """Train the deterministic decoder."""
        # Get latent representations
        with torch.no_grad():
            latent_states = self.state_embedder(states)
            latent_next_states = self.state_embedder(next_states)
            latent_actions = self.latent_action_generator(latent_states, latent_next_states)
        
        # Compute loss
        loss = self.decoder.loss(latent_states, latent_actions, latent_next_states, self.state_embedder)
        
        # Backpropagation
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
        
        return loss.item()
    
    def train_step(self):
        """Execute a single training step."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        # Sample transitions from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Train the embedder
        embedder_loss = self.train_embedder(states)
        
        # Train the latent action generator
        lag_loss = self.train_latent_action_generator(states, next_states)
        
        # Train the decoder
        decoder_loss = self.train_decoder(states, actions, next_states)
        
        return {
            'embedder_loss': embedder_loss,
            'lag_loss': lag_loss,
            'decoder_loss': decoder_loss
        }
    
    def save_checkpoint(self, step):
        """Save model checkpoints."""
        checkpoint = {
            'state_embedder': self.state_embedder.state_dict(),
            'latent_action_generator': self.latent_action_generator.state_dict(),
            'decoder': self.decoder.state_dict(),
            'embedder_optimizer': self.embedder_optimizer.state_dict(),
            'lag_optimizer': self.lag_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'metrics': self.metrics,
            'step': step
        }
        
        torch.save(checkpoint, os.path.join(
            self.config.checkpoint_dir, f"checkpoint_{step}.pt"))
    
    def load_checkpoint(self, path):
        """Load model checkpoints."""
        checkpoint = torch.load(path)
        
        self.state_embedder.load_state_dict(checkpoint['state_embedder'])
        self.latent_action_generator.load_state_dict(checkpoint['latent_action_generator'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        self.embedder_optimizer.load_state_dict(checkpoint['embedder_optimizer'])
        self.lag_optimizer.load_state_dict(checkpoint['lag_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        
        self.metrics = checkpoint['metrics']
        
        return checkpoint['step']
    
    def evaluate(self, num_episodes=3):
        """Evaluate the current policy."""
        eval_env = self.create_env()
        total_rewards = []
        
        for i in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.get_action(state, explore=False)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        eval_env.close()
        return np.mean(total_rewards)
    
    def collect_experience(self, num_steps):
        """Collect experience using random actions."""
        env = self.create_env()
        state, _ = env.reset()
        
        for _ in range(num_steps):
            # Take random action
            action = env.action_space.sample()
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Add to replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Reset if episode is done
            if done:
                state, _ = env.reset()
        
        env.close()
    
    def train(self):
        """Main training loop."""
        # Create environment
        self.env = self.create_env()
        
        # Fill replay buffer with initial experience
        print("Collecting initial experience...")
        self.collect_experience(self.config.min_buffer_size)
        
        print("Starting training...")
        total_steps = 0
        episode = 0
        state, _ = self.env.reset()
        episode_reward = 0
        
        # Training progress bar
        pbar = tqdm(total=self.config.total_steps)
        
        while total_steps < self.config.total_steps:
            # Early exploration phase with random actions
            if total_steps < self.config.exploration_steps:
                action = self.env.action_space.sample()
                latent_action = np.zeros(self.config.latent_dim)  # Placeholder
            else:
                # Use learned policy
                action, latent_action = self.get_action(state, explore=True)
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Add to replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train models
            if total_steps % 4 == 0:  # Update networks every 4 steps
                losses = self.train_step()
                if losses:
                    self.metrics['embedder_losses'].append(losses['embedder_loss'])
                    self.metrics['lag_losses'].append(losses['lag_loss'])
                    self.metrics['decoder_losses'].append(losses['decoder_loss'])
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            total_steps += 1
            pbar.update(1)
            
            # End of episode
            if done:
                self.metrics['episode_rewards'].append(episode_reward)
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
                episode += 1
                episode_reward = 0
                state, _ = self.env.reset()
            
            # Periodic evaluation
            if total_steps % self.config.eval_interval == 0:
                eval_reward = self.evaluate(self.config.eval_episodes)
                self.metrics['eval_rewards'].append(eval_reward)
                print(f"Step {total_steps}, Eval reward: {eval_reward:.2f}")
            
            # Save checkpoint
            if total_steps % self.config.checkpoint_interval == 0:
                self.save_checkpoint(total_steps)
        
        # Final checkpoint and evaluation
        self.save_checkpoint(total_steps)
        final_eval = self.evaluate(10)  # More thorough final evaluation
        print(f"Training completed. Final evaluation: {final_eval:.2f}")
        
        # Close environment
        self.env.close()
        pbar.close()
        
        return self.metrics
    
    def visualize_latent_space(self, num_samples=1000):
        """Visualize the learned latent space using t-SNE."""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
        except ImportError:
            print("sklearn and matplotlib are required for visualization")
            return
        
        # Collect samples
        env = self.create_env()
        states = []
        
        state, _ = env.reset()
        for _ in range(num_samples):
            states.append(state)
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = env.reset()
        
        env.close()
        
        # Get latent representations
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        with torch.no_grad():
            latent_states = self.state_embedder(states_tensor).cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_states)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
        plt.title("t-SNE Visualization of Latent Space")
        plt.savefig("latent_space_visualization.png")
        plt.close()
        
        print("Latent space visualization saved to latent_space_visualization.png")
    
    def render_policy(self, num_episodes=1, save_video=False):
        """Render the policy in action."""
        if save_video:
            env = gym.make(self.config.env_name, continuous=True, render_mode="rgb_array")
            env = RecordVideo(env, "videos/", episode_trigger=lambda e: True)
        else:
            env = gym.make(self.config.env_name, continuous=True, render_mode="human")
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.get_action(state, explore=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        env.close()


# =============================================================================
# MAIN SCRIPT TO RUN THE TRAINING
# =============================================================================

def main():
    """Main function to run the training."""
    config = Config()
    
    # Create agent
    agent = LatentActionAgent(config)
    
    # Train agent
    metrics = agent.train()
    
    # Visualize results
    agent.visualize_latent_space()
    
    # Render trained policy
    agent.render_policy(num_episodes=3, save_video=True)
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['eval_rewards'])
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['embedder_losses'], label='Embedder')
    plt.plot(metrics['lag_losses'], label='Latent Action Generator')
    plt.plot(metrics['decoder_losses'], label='Decoder')
    plt.title('Training Losses')
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    print("Training metrics saved to training_metrics.png")


if __name__ == "__main__":
    main()