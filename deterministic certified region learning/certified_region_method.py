import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Hyperparameters
# ---------------------------
LATENT_DIM = 8
HIDDEN_DIM = 16
BATCH_SIZE = 16
GAMMA = 0.9
LEARNING_RATE = 1e-3
MEMORY_CAPACITY = 1000
EPISODES = 100
MAX_STEPS = 200
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
CERT_EPSILON = 0.1   # tolerance for certification (allowed Q-value deviation)
CONTROLLER_THRESHOLD = 1.0  # TD error threshold for adjusting certification
CONTROLLER_ALPHA = 0.9  # factor to reduce radius if error is high
CONTROLLER_BETA = 1.1   # factor to increase radius if error is low
MIN_RADIUS = 0.01
MAX_RADIUS = 1.0

# ---------------------------
# Encoder Network: Maps state to latent space.
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ---------------------------
# QNetwork (Generalizer): Estimates Q-values from latent representation.
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_actions=3):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, num_actions)
        )
        
    def forward(self, z):
        return self.net(z)

# ---------------------------
# Replay Buffer for Experience Replay.
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
        
    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Memory Module: Stores latent samples with certified radius.
# ---------------------------
class MemoryModule:
    def __init__(self):
        # Each entry is a dict with keys: 'z', 'radius', 'action', 'q_value'
        self.memory = []
        
    def add(self, z, action, q_value, radius):
        self.memory.append({
            'z': z, 
            'radius': radius, 
            'action': action, 
            'q_value': q_value
        })
        
    def update_radius(self, td_error):
        # Adjust each memory sample's radius based on the observed TD error.
        for entry in self.memory:
            if td_error > CONTROLLER_THRESHOLD:
                entry['radius'] = max(MIN_RADIUS, entry['radius'] * CONTROLLER_ALPHA)
            else:
                entry['radius'] = min(MAX_RADIUS, entry['radius'] * CONTROLLER_BETA)
                
    def get_certified_q(self, z):
        # Check if latent vector z lies within any certified ball; return average Q-value if so.
        certified_qs = []
        for entry in self.memory:
            if np.linalg.norm(z - entry['z']) <= entry['radius']:
                certified_qs.append(entry['q_value'])
        if len(certified_qs) > 0:
            return np.mean(certified_qs)
        else:
            return None
        
    def coverage(self, z):
        # Returns True if z is covered by any certified region.
        for entry in self.memory:
            if np.linalg.norm(z - entry['z']) <= entry['radius']:
                return True
        return False

# ---------------------------
# Compute Certified Radius: Binary search to determine maximal radius
# such that perturbations in the latent space do not change the Q-value (for the given action)
# by more than CERT_EPSILON.
# ---------------------------
def compute_certified_radius(encoder, q_network, z, action, epsilon=CERT_EPSILON, 
                             min_r=MIN_RADIUS, max_r=MAX_RADIUS, num_samples=10, tol=1e-3):
    low = min_r
    high = max_r
    best_r = low
    while high - low > tol:
        mid = (low + high) / 2.0
        # Sample random perturbations in the latent space ball of radius 'mid'
        deltas = np.random.randn(num_samples, z.shape[0])
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        deltas = deltas / (norms + 1e-6)  # unit vectors
        scales = np.random.uniform(0, mid, size=(num_samples, 1))
        perturbations = deltas * scales
        z_perturbed = z + perturbations  # shape: (num_samples, latent_dim)
        z_tensor = torch.tensor(z_perturbed, dtype=torch.float32, device=device)
        q_vals = q_network(z_tensor)  # shape: (num_samples, num_actions)
        q_vals = q_vals[:, action].detach().cpu().numpy()
        # Compute Q-value at the original latent vector z for the given action.
        z_orig_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        q_orig = q_network(z_orig_tensor)[0, action].item()
        differences = np.abs(q_vals - q_orig)
        if np.max(differences) <= epsilon:
            best_r = mid
            low = mid  # Try expanding the radius.
        else:
            high = mid  # Decrease the radius.
    return best_r

# ---------------------------
# DQN Agent integrating Memory, Generalizer, and Controller.
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.encoder = Encoder(input_dim=state_dim).to(device)
        self.q_network = QNetwork(latent_dim=LATENT_DIM, num_actions=action_dim).to(device)
        self.target_q_network = QNetwork(latent_dim=LATENT_DIM, num_actions=action_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + 
                                    list(self.q_network.parameters()), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()
        self.memory_module = MemoryModule()
        self.epsilon = EPS_START
        self.action_dim = action_dim
        
    def select_action(self, state):
        # Convert state to tensor and encode into latent space.
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        z = self.encoder(state_tensor).detach().cpu().numpy().squeeze()
        # If a certified Q-value exists in memory for this latent state, use it (if not exploring).
        certified_q = self.memory_module.get_certified_q(z)
        if certified_q is not None and np.random.rand() > self.epsilon:
            # Use the Q-network prediction here as the certified Q is a scalar; we pick the best action.
            with torch.no_grad():
                q_vals = self.q_network(self.encoder(state_tensor))
            action = q_vals.argmax().item()
        else:
            # Epsilon-greedy action selection.
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    q_vals = self.q_network(self.encoder(state_tensor))
                action = q_vals.argmax().item()
        return action
    
    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        # Encode current states
        z_states = self.encoder(states)
        q_values = self.q_network(z_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using the target network.
        with torch.no_grad():
            z_next = self.encoder(next_states)
            next_q_values = self.target_q_network(z_next)
            next_q_max, _ = next_q_values.max(dim=1)
            target_q = rewards + GAMMA * next_q_max * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute TD errors for the batch.
        td_errors = torch.abs(q_values - target_q).detach().cpu().numpy()
        # For each sample in the batch, update the memory module.
        for i in range(BATCH_SIZE):
            z = z_states[i].detach().cpu().numpy()
            action = actions[i].item()
            q_val = q_values[i].item()
            # Compute certified radius using the current encoder and Q-network.
            radius = compute_certified_radius(self.encoder, self.q_network, z, action)
            self.memory_module.add(z, action, q_val, radius)
        
        # Use average TD error from the batch to update memory radii via the Controller.
        avg_td_error = np.mean(td_errors)
        self.memory_module.update_radius(avg_td_error)
        
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

# ---------------------------
# Main Training Loop
# ---------------------------
def train():
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]  # Typically 2: position and velocity.
    action_dim = env.action_space.n              # Typically 3 actions.
    agent = DQNAgent(state_dim, action_dim)
    
    scores = []
    for episode in tqdm(range(EPISODES)):
        state, _ = env.reset()
        total_reward = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # In MountainCar-v0, reward is -1 per timestep until the goal is reached.
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            agent.update()
            if done or truncated:
                break
        agent.update_target_network()
        agent.decay_epsilon()
        scores.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    env.close()
    return scores

if __name__ == "__main__":
    scores = train()
