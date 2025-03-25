import numpy as np

def policy(current_inv, theta_min, theta_max):
    if current_inv < theta_min:
        return theta_max - current_inv
    else:
        return 0

def transition(current_inv, order_amt, demand):
    return max(current_inv + order_amt - demand, 0)

def reward(current_inv, order_amt, demand, price, cost):
    sales = min(current_inv + order_amt, demand)
    return price * sales - cost * order_amt

# Simulate one episode of 200 steps
def simulate_episode(theta_min, theta_max, price, cost, lambda_, steps=200, init_inv=0):
    current_inv = init_inv  # Initial inventory
    total_profit = 0
    for _ in range(steps):
        order_amt = policy(current_inv, theta_min, theta_max)              # Decide order quantity
        demand = np.random.poisson(lambda_)   # Sample demand
        profit = reward(current_inv, order_amt, demand, price, cost)   # Compute profit
        total_profit += profit
        current_inv = transition(current_inv, order_amt, demand)          # Update inventory
    return total_profit

def evaluate_policy(theta_min, theta_max, price, cost, lambda_, num_episodes=100, steps=200):
    total_profits = [simulate_episode(theta_min, theta_max, price, cost, lambda_, steps) for _ in range(num_episodes)]
    return np.mean(total_profits)

# Parameters
price = 10          # Selling price per unit
cost = 5           # Cost per unit
lambda_ = 10    # Demand distribution parameter (Poisson mean)
steps = 200         # Steps per episode
num_episodes = 100  # Number of episodes for evaluation

# Grid search to find optimal policy
best_profit = -np.inf
best_theta_min = None
best_theta_max = None

# Search over theta_min from 0 to 15 and theta_max from theta_min to 25
for theta_min in range(0, 16):
    for theta_max in range(theta_min, 26):
        avg_profit = evaluate_policy(theta_min, theta_max, price, cost, lambda_, num_episodes, steps)
        if avg_profit > best_profit:
            best_profit = avg_profit
            best_theta_min = theta_min
            best_theta_max = theta_max
            print(f"New best: theta_min={theta_min}, theta_max={theta_max}, average profit={avg_profit:.2f}")

# Print the optimal policy and its performance
print(f"\nOptimal policy found:")
print(f"s (θ_min) = {best_theta_min}, S (θ_max) = {best_theta_max}")
print(f"Average total profit over {num_episodes} episodes of {steps} steps: {best_profit:.2f}")