# --- Imports ---
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import gymnasium_2048.envs
import time

# Model without Priority Replay Buffer
# --- DQN Model Definition ---
class DQN(nn.Module):
    def __init__(self, input_dim=16, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Utility Functions ---
def preprocess(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    return np.log2(state + 1).flatten() / 16.0

def find_high_tile(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    return np.max(state)

def shape_reward(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    bonus = 0
    grid = np.array(state).tolist()

    for row in grid:
        if all(float(x) <= float(y) for x, y in zip(row, row[1:])):
            bonus += 1
        if all(float(x) >= float(y) for x, y in zip(row, row[1:])):
            bonus += 1

    for col in zip(*grid):
        if all(float(x) <= float(y) for x, y in zip(col, col[1:])):
            bonus += 1
        if all(float(x) >= float(y) for x, y in zip(col, col[1:])):
            bonus += 1

    bonus += np.log2(grid[3][3] + 1)
    bonus += np.log2(grid[3][2] + 1) / 2
    bonus += np.log2(grid[3][1] + 1) / 4
    bonus += np.log2(grid[3][0] + 1) / 8

    return bonus * 0.1

# --- Main Training ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

    main_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters(), lr=1e-4)

    buffer = deque(maxlen=10000)

    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10
    num_episodes = 500
    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        high_tile = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    action = main_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.array_equal(state, next_state):
                shaped_reward = -100
                reward = -100
            else:
                bonus = shape_reward(next_state)
                shaped_reward = reward + bonus

            done = terminated or truncated
            buffer.append((state, action, shaped_reward, next_state, done))

            state = next_state
            high_tile = max(high_tile, find_high_tile(state))
            total_reward += reward + high_tile

            if len(buffer) >= batch_size:
                samples = random.sample(buffer, batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*samples)

                batch_states = torch.tensor(np.array([preprocess(s) for s in batch_states]), dtype=torch.float32).to(device)
                batch_actions = torch.tensor(batch_actions).to(device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_states = torch.tensor(np.array([preprocess(s) for s in batch_next_states]), dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)

                q_values = main_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                next_actions = main_net(batch_next_states).argmax(1, keepdim=True)
                next_q_values = target_net(batch_next_states).gather(1, next_actions).squeeze(1)

                target_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)

                loss = F.mse_loss(q_values, target_q_values.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps += 1
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 10 == 0:
            print(f"Episode {episode:03d} | Score: {total_reward:.0f} | High tile: {high_tile} | Epsilon: {epsilon:.3f}")

    env.close()
    torch.save(main_net.state_dict(), "dqn_2048.pth")
    print("Model saved!")
