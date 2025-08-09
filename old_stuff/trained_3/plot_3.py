import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import gymnasium_2048.envs
import time
import csv

from trained_3.model_3 import DQN, preprocess, find_high_tile

def test_2048():
    # Write header to results file
    with open('results_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["high_tile"])

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GOALS =  [64, 128, 256, 512, 1024]

    meta_model = DQN(input_dim=16, output_dim=len(GOALS)).to(device)
    meta_model.load_state_dict(torch.load("meta_controller.pth", map_location=device))
    meta_model.eval()

    controller_model = DQN(input_dim=16 + len(GOALS), output_dim=4).to(device)
    controller_model.load_state_dict(torch.load("controller.pth", map_location=device))
    controller_model.eval()

    def select_goal(meta_model, state_tensor):
        with torch.no_grad():
            goal_qs = meta_model(state_tensor)
            goal_idx = torch.argmax(goal_qs).item()
        return GOALS[goal_idx]

    def goal_achieved(state, goal):
        return find_high_tile(state) >= goal

    def one_hot_goal(goal, goal_space):
        vec = np.zeros(len(goal_space), dtype=np.float32)
        if goal in goal_space:
            vec[goal_space.index(goal)] = 1.0
        return vec

    # --- Run Episodes ---
    for episode in range(1000):
        env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_moves = 0

        # Pick initial goal
        goal = select_goal(meta_model, torch.tensor([preprocess(state)], dtype=torch.float32).to(device))
        goal_vec = one_hot_goal(goal, GOALS)

        while not done:
            # Format state and goal
            state_tensor = torch.tensor([preprocess(state)], dtype=torch.float32).to(device)
            goal_tensor = torch.tensor([goal_vec], dtype=torch.float32).to(device)
            state_goal_tensor = torch.cat([state_tensor, goal_tensor], dim=1)

            # Choose action
            with torch.no_grad():
                q_vals = controller_model(state_goal_tensor).squeeze(0)
                actions = torch.argsort(q_vals, descending=True).tolist()

            for action in actions:
                next_state, reward, terminated, truncated, _ = env.step(action)
                if not np.array_equal(state, next_state):
                    break

            done = terminated or truncated
            total_reward += reward
            total_moves += 1
            state = next_state

            # Update goal if achieved
            if goal_achieved(state, goal):
                goal = select_goal(meta_model, torch.tensor([preprocess(state)], dtype=torch.float32).to(device))
                goal_vec = one_hot_goal(goal, GOALS)

        with open('results_3.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([find_high_tile(state)])

        env.close()

# Run test
test_2048()

# --- Plot Results ---
df = pd.read_csv('results_3.csv')
high_tiles = df['high_tile']
tile_values = [32, 64, 128, 256, 512, 1024, 2048]
tile_counts = {val: 0 for val in tile_values}

for tile in high_tiles:
    if tile in tile_counts:
        tile_counts[tile] += 1

# Plot
x = list(tile_counts.keys())
y = list(tile_counts.values())

plt.figure(figsize=(8, 5))
plt.bar([str(val) for val in x], y, color='skyblue', edgecolor='black')
plt.xlabel('Highest Tile Achieved')
plt.ylabel('Count')
plt.title('Distribution of Highest Tiles over 1000 Games')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Tile Frequencies:", tile_counts)
