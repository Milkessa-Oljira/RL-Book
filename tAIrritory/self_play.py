import torch
import numpy as np
from Game_Env import TAIrritoryEnv  # Ensure this file is in the same directory or correctly imported
from rl_agent import TAIrritoryAgent  # Ensure this imports your RL agent

def self_play(agent, env, num_games=10):
    """Runs self-play games between the AI playing as both players."""
    for game_idx in range(num_games):
        print(f"Starting game {game_idx + 1}/{num_games}")
        state, _ = env.reset()
        done = False
        turn = 0
        total_reward = 0

        while not done:
            action = agent.get_action(state, env, training=True)
            if action is None:
                print(f"No valid actions for player {env.current_player}.")
                break
            
            # Step the environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Record the transition (state, action, reward)
            agent.replay_buffer.push(state, action, reward)

            # Train after every move
            agent.train()

            turn += 1
            if turn > 100:  # Safety cutoff for infinite games
                print("Game reached turn limit. Ending...")
                break

        # Game over, save model
        print(f"Game {game_idx + 1} finished. Reward: {total_reward}")
        agent.save()

def main():
    env = TAIrritoryEnv()
    agent = TAIrritoryAgent()

    # Load the model if it exists
    try:
        agent.load()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}. Starting with a new model.")

    # Run self-play games
    self_play(agent, env, num_games=10)

if __name__ == "__main__":
    main()
