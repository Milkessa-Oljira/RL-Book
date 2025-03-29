import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Hyperparameters and Utility Functions
#############################################

# Environment: CarRacing-v2 (high-dim observation, low-dim action)
ENV_NAME = "CarRacing-v3"
d_obs = 96 * 96 * 3        # Observation: 96x96 RGB image flattened
d_action = 3               # Action: 3-dimensional continuous
latent_dim = int(np.sqrt(d_obs * d_action))  # = sqrt(27,648*3) ≈ 288

# For the transformer: reshape latent vector into tokens.
n_tokens = 16
token_dim = latent_dim // n_tokens  # e.g., 288 // 16 = 18

# Learning rates and consolidation parameters
eta_fast = 1e-3
eta_slow = 1e-5   # via EMA
tau = 0.99        # EMA coefficient for slow weights update
lambda_c = 1e-2   # consolidation weight for fast parameters vs. slow parameters
lambda_reward = 0.1  # reward-based energy weight

# Epsilon for e-greedy action selection
epsilon = 0.3

# Threshold for using memory prediction
memory_threshold = 0.8

# Transform: Normalize observations to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # converts HWC [0,255] to CHW [0,1]
])

#############################################
# Define the Network Components
#############################################

class ObservationEncoder(nn.Module):
    """CNN encoder mapping an image observation to a latent vector."""
    def __init__(self, latent_dim):
        super(ObservationEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, 8, 8)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # x shape: (batch, 3, 96, 96)
        x = F.interpolate(x, size=(96, 96))  # ensure correct size
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent

class ActionEncoder(nn.Module):
    """MLP encoder mapping a continuous action (dim=3) to a latent vector."""
    def __init__(self, latent_dim):
        super(ActionEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_action, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, a):
        return self.net(a)

class ActionDecoder(nn.Module):
    """Decoder that maps a latent action back to the original action space."""
    def __init__(self, latent_dim):
        super(ActionDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, d_action)
        )

    def forward(self, latent_act):
        # Optionally, add a tanh activation if actions are bounded in [-1,1]
        return self.net(latent_act)

class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor: given the latent observation (reshaped into tokens)
    predicts the latent action. Contains both fast (generalizing) and slow (associative memory)
    parameters updated via EMA.
    """
    def __init__(self, latent_dim, n_tokens, token_dim, nhead=2, num_layers=2):
        super(TransformerPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        # Project latent observation to token space
        self.token_proj = nn.Linear(latent_dim, latent_dim)

        # Transformer Encoder for fast weights
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead)
        self.transformer_fast = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection from tokens back to latent vector (fast path)
        self.token_reconstruct_fast = nn.Linear(latent_dim, latent_dim)

        # Gating network: from latent observation to a scalar gate in [0,1]
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Create a slow copy of the transformer and projection layers
        self.transformer_slow = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.token_reconstruct_slow = nn.Linear(latent_dim, latent_dim)
        self._initialize_slow_parameters()

    def _initialize_slow_parameters(self):
        self.transformer_slow.load_state_dict(self.transformer_fast.state_dict())
        self.token_reconstruct_slow.load_state_dict(self.token_reconstruct_fast.state_dict())

    def forward(self, latent_obs):
        """
        latent_obs: (batch, latent_dim)
        Returns:
            pred_latent: predicted latent action via gated combination of fast and slow outputs.
            gate: the gating value.
        """
        batch_size = latent_obs.size(0)
        # Project and reshape to tokens: (batch, n_tokens, token_dim)
        tokens = self.token_proj(latent_obs)  # shape: (batch, latent_dim)
        tokens = tokens.view(batch_size, self.n_tokens, self.token_dim)

        # Permute tokens for transformer input: (n_tokens, batch, token_dim)
        tokens = tokens.permute(1, 0, 2)

        # Fast pathway
        out_fast = self.transformer_fast(tokens)  # (n_tokens, batch, token_dim)
        out_fast = out_fast.permute(1, 0, 2).contiguous().view(batch_size, self.latent_dim)
        pred_fast = self.token_reconstruct_fast(out_fast)

        # Slow pathway (memory)
        out_slow = self.transformer_slow(tokens)
        out_slow = out_slow.permute(1, 0, 2).contiguous().view(batch_size, self.latent_dim)
        pred_slow = self.token_reconstruct_slow(out_slow)

        # Gate to combine fast and slow predictions
        gate = self.gate_net(latent_obs)  # shape: (batch, 1)
        pred_latent = gate * pred_slow + (1 - gate) * pred_fast
        return pred_latent, gate

    def update_slow_parameters(self):
        # EMA update for slow parameters
        for param_slow, param_fast in zip(self.transformer_slow.parameters(), self.transformer_fast.parameters()):
            param_slow.data = tau * param_slow.data + (1 - tau) * param_fast.data
        for param_slow, param_fast in zip(self.token_reconstruct_slow.parameters(), self.token_reconstruct_fast.parameters()):
            param_slow.data = tau * param_slow.data + (1 - tau) * param_fast.data

class ActionTransformerPredictor(nn.Module):
    """
    Transformer-based predictor within the JEPA architecture to directly predict
    the latent action from the latent observation.
    """
    def __init__(self, latent_dim, n_tokens, token_dim, nhead=2, num_layers=2):
        super(ActionTransformerPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        # Project latent observation to token space
        self.token_proj = nn.Linear(latent_dim, latent_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection from tokens back to latent action vector
        self.token_reconstruct = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent_obs):
        """
        latent_obs: (batch, latent_dim)
        Returns:
            pred_latent: predicted latent action.
        """
        batch_size = latent_obs.size(0)
        # Project and reshape to tokens: (batch, n_tokens, token_dim)
        tokens = self.token_proj(latent_obs)  # shape: (batch, latent_dim)
        tokens = tokens.view(batch_size, self.n_tokens, self.token_dim)

        # Permute tokens for transformer input: (n_tokens, batch, token_dim)
        tokens = tokens.permute(1, 0, 2)

        # Transformer
        out = self.transformer(tokens)  # (n_tokens, batch, token_dim)
        out = out.permute(1, 0, 2).contiguous().view(batch_size, self.latent_dim)
        pred_latent = self.token_reconstruct(out)
        return pred_latent

class JEPA_Agent(nn.Module):
    def __init__(self, latent_dim, n_tokens, token_dim):
        super(JEPA_Agent, self).__init__()
        self.obs_encoder = ObservationEncoder(latent_dim).to(device)
        self.act_encoder = ActionEncoder(latent_dim).to(device)
        self.act_decoder = ActionDecoder(latent_dim).to(device)
        self.memory_predictor = TransformerPredictor(latent_dim, n_tokens, token_dim).to(device)
        self.action_predictor = ActionTransformerPredictor(latent_dim, n_tokens, token_dim).to(device)

    def forward(self, obs):
        # obs: (B, 3, 96, 96)
        latent_obs = self.obs_encoder(obs)
        pred_latent_action = self.action_predictor(latent_obs)
        return latent_obs, pred_latent_action

#############################################
# Energy Function (Reward Regularization)
#############################################

def energy_function(latent_obs, latent_action, pred_latent, reward):
    """
    Computes energy based on cosine similarity between the predicted and actual latent actions,
    weighted by the reward. Higher reward should yield lower energy.
    """
    cos_sim = F.cosine_similarity(pred_latent, latent_action, dim=1)
    energy = -reward * cos_sim  # high reward => lower energy
    return energy.mean()

#############################################
# Training Loop on Gymnasium Environment
#############################################

def train_agent(num_episodes=500, max_steps=1000):
    env = gym.make(ENV_NAME, render_mode=None)
    agent = JEPA_Agent(latent_dim, n_tokens, token_dim)

    # Optimizer for all fast parameters of both transformers and encoders/decoders
    optimizer = optim.Adam([
        {'params': agent.obs_encoder.parameters()},
        {'params': agent.act_encoder.parameters()},
        {'params': agent.act_decoder.parameters()},
        {'params': agent.memory_predictor.token_proj.parameters()},
        {'params': agent.memory_predictor.transformer_fast.parameters()},
        {'params': agent.memory_predictor.token_reconstruct_fast.parameters()},
        {'params': agent.memory_predictor.gate_net.parameters()},
        {'params': agent.action_predictor.token_proj.parameters()},
        {'params': agent.action_predictor.transformer.parameters()},
        {'params': agent.action_predictor.token_reconstruct.parameters()},
    ], lr=eta_fast)

    replay_buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            # Preprocess observation
            obs_tensor = transform(obs).unsqueeze(0).to(device)  # shape: (1, 3, 96, 96)
            with torch.no_grad():
                latent_obs = agent.obs_encoder(obs_tensor)

            # Action selection logic
            with torch.no_grad():
                # Check memory prediction
                pred_latent_memory, gate_memory = agent.memory_predictor(latent_obs)

                if gate_memory.item() > memory_threshold:
                    # Use memory prediction
                    latent_action_pred = pred_latent_memory
                else:
                    # Use JEPA's action predictor
                    latent_action_pred = agent.action_predictor(latent_obs)

                # Decode latent action to actual action using the action decoder
                action_tensor = agent.act_decoder(latent_action_pred)
                action = action_tensor.squeeze(0).cpu().numpy()

            action = np.clip(action, -1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition in replay buffer
            replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs

            # Training starts after collecting a sufficient number of samples
            if len(replay_buffer) > 32:
                batch = [replay_buffer[np.random.randint(len(replay_buffer))] for _ in range(32)]
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

                obs_batch = torch.stack([transform(o) for o in obs_batch]).to(device)
                action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(device)
                reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)

                # Forward pass through agent
                latent_obs_batch, pred_latent_action_batch = agent(obs_batch)
                latent_action_batch = agent.act_encoder(action_batch) # Encode the actual action

                # Predictive loss in latent space (MSE)
                loss_predict = F.mse_loss(pred_latent_action_batch, latent_action_batch)

                # Reward-based energy regularization loss
                loss_energy = energy_function(latent_obs_batch, latent_action_batch, pred_latent_action_batch, reward_batch)

                total_loss = loss_predict + lambda_reward * loss_energy

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Consolidation for memory predictor
                loss_consolidation_memory = 0.0
                fast_params_memory = list(agent.memory_predictor.transformer_fast.parameters()) + \
                                    list(agent.memory_predictor.token_reconstruct_fast.parameters())
                slow_params_memory = list(agent.memory_predictor.transformer_slow.parameters()) + \
                                    list(agent.memory_predictor.token_reconstruct_slow.parameters())
                for p_fast, p_slow in zip(fast_params_memory, slow_params_memory):
                    loss_consolidation_memory += F.mse_loss(p_fast, p_slow)
                loss_consolidation_memory = lambda_c * loss_consolidation_memory
                loss_consolidation_memory.backward()
                optimizer.step()

                # Update slow parameters of memory predictor using EMA
                agent.memory_predictor.update_slow_parameters()

                # Consolidation for action predictor (assuming it also has fast/slow weights - for simplicity, we'll add slow weights now)
                if not hasattr(agent.action_predictor, 'transformer_slow'):
                    agent.action_predictor.transformer_slow = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=token_dim, nhead=2), num_layers=2
                    ).to(device)
                    agent.action_predictor.token_reconstruct_slow = nn.Linear(latent_dim, latent_dim).to(device)
                    agent.action_predictor.transformer_slow.load_state_dict(agent.action_predictor.transformer.state_dict())
                    agent.action_predictor.token_reconstruct_slow.load_state_dict(agent.action_predictor.token_reconstruct.state_dict())

                loss_consolidation_action = 0.0
                fast_params_action = list(agent.action_predictor.transformer.parameters()) + \
                                    list(agent.action_predictor.token_reconstruct.parameters())
                slow_params_action = list(agent.action_predictor.transformer_slow.parameters()) + \
                                    list(agent.action_predictor.token_reconstruct_slow.parameters())
                for p_fast, p_slow in zip(fast_params_action, slow_params_action):
                    loss_consolidation_action += F.mse_loss(p_fast, p_slow)
                loss_consolidation_action = lambda_c * loss_consolidation_action
                loss_consolidation_action.backward()
                optimizer.step()

                # Update slow parameters of action predictor using EMA
                for param_slow, param_fast in zip(agent.action_predictor.transformer_slow.parameters(), agent.action_predictor.transformer.parameters()):
                    param_slow.data = tau * param_slow.data + (1 - tau) * param_fast.data
                for param_slow, param_fast in zip(agent.action_predictor.token_reconstruct_slow.parameters(), agent.action_predictor.token_reconstruct.parameters()):
                    param_slow.data = tau * param_slow.data + (1 - tau) * param_fast.data

            if done:
                break
        print(f"Episode {episode+1}/{num_episodes} Reward: {episode_reward:.2f}")
    env.close()

#############################################
# Run Training
#############################################

if __name__ == "__main__":
    train_agent(num_episodes=100)