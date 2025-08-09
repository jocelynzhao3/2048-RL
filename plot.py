"""
Evaluate and plot results for trained model
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

import torch
import numpy as np
import gymnasium as gym
import gymnasium_2048.envs

from model import DQN_CNN, preprocess, find_high_tile
import csv
from torchsummary import summary


def test_2048():

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN_CNN().to(device)
    model.load_state_dict(torch.load("trained_models/dqn_2048_3000.pth", map_location=device))
    model.eval()

    with open('results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "moves", "score", "high_tile"])

    for i in range(1000):

        # Play one episode
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        total_moves = 0

        while not done:
            state_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor).squeeze(0)  # shape: [4]

                actions = torch.argsort(q_vals, descending=True).tolist()
                for action in actions:
                    if action == 2:
                        continue
                    next_state, reward, terminated, truncated, _ = env.step(action) # 1 is right, 0 is up, 3 is left, 2 is down
                    if not np.array_equal(state, next_state):
                        break
                else:
                    next_state, reward, terminated, truncated, _ = env.step(2)

            done = terminated or truncated
            total_reward += reward
            total_moves += 1
            state = next_state

        grid = np.argmax(state, axis=2)
        grid = 2 ** grid * (grid > 0)

        with open('results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, total_moves, total_reward, find_high_tile(state)])

    env.close()

def baseline_2048():

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

    with open('baseline.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "moves", "score", "high_tile"])

    for i in range(1000):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        total_moves = 0

        while not done:
            action = random.randint(0, 3)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_moves += 1
            state = next_state

        grid = np.argmax(state, axis=2)
        grid = 2 ** grid * (grid > 0)
        high_tile = np.max(grid)

        with open('baseline.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, total_moves, total_reward, high_tile])

    env.close()

def eval():
    last_x_runs = 50
    tile_thresholds = [128, 256, 512, 1024]

    # --- LOAD CSV ---
    df = pd.read_csv("logs/run.csv")[:500]
    df = df.sort_values("episode")

    # --- PLOT ---
    plt.figure(figsize=(10, 5))

    for tile in tile_thresholds:
        rolling_hits = (
            df["high_tile"].rolling(last_x_runs)
            .apply(lambda window: (window >= tile).mean() * 100, raw=False)
        )
        plt.plot(df["episode"], rolling_hits, marker='.', label=f"≥ {tile} tile")

    plt.xlabel("Episode")
    plt.ylabel(f"Percent of last {last_x_runs} runs ≥ tile")
    plt.title(f"Consistency Trends (last {last_x_runs} runs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(DQN_CNN().to(device), (16, 4, 4))

    print("Running baseline (random action) model...")
    baseline_2048()
    print("Running trained model...")
    test_2048()

    df = pd.read_csv('results.csv')
    baselines_df = pd.read_csv('baseline.csv')

    high_tiles_agent = df['high_tile']
    high_tiles_baseline = baselines_df['high_tile']

    tile_values = [32, 64, 128, 256, 512, 1024, 2048]

    agent_counts = {val: 0 for val in tile_values}
    for tile in high_tiles_agent:
        if tile in agent_counts:
            agent_counts[tile] += 1

    baseline_counts = {val: 0 for val in tile_values}
    for tile in high_tiles_baseline:
        if tile in baseline_counts:
            baseline_counts[tile] += 1

    x = np.arange(len(tile_values))
    y_agent = list(agent_counts.values())
    y_baseline = list(baseline_counts.values())

    bar_width = 0.45

    plt.figure(figsize=(8, 5))
    plt.bar(x + bar_width/2, y_agent, width=bar_width, label='Agent', color='skyblue', edgecolor='black')
    plt.bar(x - bar_width/2, y_baseline, width=bar_width, label='Baseline', color='salmon', edgecolor='black')

    plt.xticks(x, [str(val) for val in tile_values])
    plt.xlabel('Highest Tile Achieved')
    plt.ylabel('Count')
    plt.title('Distribution of Highest Tiles: Agent vs Baseline')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f'Trained model results in 1000 games: {agent_counts}')
