import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class LatentSpaceEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(LatentSpaceEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

class LatentSpaceModel(nn.Module):
    def __init__(self, latent_dim, output_dim):  # Output dim depends on what you want to predict
        super(LatentSpaceModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


class MountainCarDeterministic:
    def __init__(self, epsilon=0.05, latent_dim=4, neighbor_radius=0.2, max_steps=200, learning_rate=0.001):
        self.env = gym.make('MountainCar-v0')
        self.epsilon = epsilon
        self.latent_dim = latent_dim
        self.neighbor_radius = neighbor_radius
        self.max_steps = max_steps

        self.phi = LatentSpaceEmbedding(self.env.observation_space.shape[0], 1, latent_dim)  # Now a neural network
        self.f = LatentSpaceModel(latent_dim, latent_dim) # Predict next latent state
        self.phi_optimizer = optim.Adam(self.phi.parameters(), lr=learning_rate)
        self.f_optimizer = optim.Adam(self.f.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

        self.D = []  # Dataset of certified points
        self.C = set() # Set of certified regions
        self.knn = NearestNeighbors(radius=neighbor_radius)

    def _compute_r(self, z):
        if not self.D:
            return 0

        # CRUCIAL FIX: Reshape BEFORE creating the list for KNN
        points_for_knn = np.array([point.reshape(-1) for point, _ in self.D])
        self.knn.fit(points_for_knn)

        z_np = z.detach().numpy().reshape(1, -1)  # Reshape z to (1, latent_dim)

        if any(np.allclose(z_np, point) for point, _ in self.D):
            certified_z_radius = next(radius for point, radius in self.D if np.allclose(z_np, point))
            return certified_z_radius

        neighbors_indices = self.knn.radius_neighbors(z_np, return_distance=False)[0]
        neighbors = [self.D[i][0] for i in neighbors_indices]

        if not neighbors:
            return 0

        max_radius = 0
        for neighbor in neighbors:
            radius = np.linalg.norm(z_np.flatten() - neighbor.flatten())

            all_within_tolerance = True
            for _ in range(10):
                perturbation = np.random.uniform(-radius, radius, size=self.latent_dim)
                z_prime = torch.tensor(neighbor + perturbation, dtype=torch.float32)

                if torch.norm(self.f(z) - self.f(z_prime)) > self.epsilon:
                    all_within_tolerance = False
                    break

            if all_within_tolerance:
                max_radius = max(max_radius, radius)

        return max_radius

    def _update_C(self):
        self.C = set()
        for z, r in self.D:
            for i in range(20):
                angle = np.random.uniform(0, 2 * np.pi)

                # CRUCIAL FIX: Handle r=0 case separately
                if r == 0:
                    sample = z.copy()  # Just use the original z if r is 0
                else:
                    perturbation = r * np.array([np.cos(angle), np.sin(angle), 0, 0])[:self.latent_dim]
                    z_np = z.copy()
                    sample = z_np + perturbation

                # Ensure sample is 1D before converting to list
                sample = sample.reshape(-1)  # Flatten

                sample_list = [float(x) for x in sample.tolist()]
                self.C.add(tuple(sample_list))
                            
    def train(self, episodes=100):
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for t in range(self.max_steps):
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                z = self.phi(state_tensor, action_tensor)
                next_z = self.phi(next_state_tensor, action_tensor)  # Embedding of the next state

                r = self._compute_r(z)

                # CRUCIAL FIX: Store z as a 2D array (1, latent_dim) in D
                z_np = z.detach().numpy().reshape(1, -1)  # Reshape before storing
                self.D.append((z_np, r))  # Store the numpy array

                # Train the embedding and the model f
                predicted_next_z = self.f(z)
                loss = self.mse_loss(predicted_next_z, next_z) # Predict next latent state

                self.phi_optimizer.zero_grad()
                self.f_optimizer.zero_grad()
                loss.backward()
                self.phi_optimizer.step()
                self.f_optimizer.step()

                self._update_C()
                state = next_state

                if terminated or truncated:
                    break

            print(f"Episode {episode+1}: Total Reward = {total_reward}")

        self.env.close()

    def _choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        best_action = None
        max_expansion = -1  # Initialize with a negative value

        for action in range(self.env.action_space.n):  # Iterate through possible actions
            action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
            z = self.phi(state_tensor, action_tensor)

            # 1. Check if the current latent state is already certified.
            if any(np.allclose(z.detach().numpy(), point) for point, _ in self.D):
                continue # If already certified, don't explore more in that direction

            # 2. Estimate potential expansion by "looking ahead"
            potential_next_state, _, _, _, _ = self.env.step(action) # Simulate one step
            potential_next_state_tensor = torch.tensor(potential_next_state, dtype=torch.float32).unsqueeze(0)
            potential_next_z = self.phi(potential_next_state_tensor, action_tensor)

            # 3. Calculate the distance to the nearest certified point.
            if self.D:
                distances = [np.linalg.norm(z.detach().numpy() - point) for point, _ in self.D]
                distance_to_nearest = min(distances)
            else:
                distance_to_nearest = float('inf')  # No certified points yet

            # 4. Encourage exploration of less-explored regions (farthest from certified points)
            expansion_potential = distance_to_nearest

            if expansion_potential > max_expansion:
                max_expansion = expansion_potential
                best_action = action

        # If no promising action was found (all were already certified or no certified points at all), explore randomly
        if best_action is None:
            return self.env.action_space.sample()

        return best_action

    def plot_certified_region(self):
         # ... (same as before)
        if self.C:
            certified_region_np = np.array(list(self.C))
            plt.scatter(certified_region_np[:, 0], certified_region_np[:, 1], s=1, label="Certified Region")
            plt.xlabel("Position")
            plt.ylabel("Velocity")
            plt.title("Certified Region in State Space")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    agent = MountainCarDeterministic()
    agent.train(episodes=50)
    agent.plot_certified_region()