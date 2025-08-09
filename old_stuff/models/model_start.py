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

# DEFAULT MODEL
# --- DQN Model ---
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


# --- Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, transition, td_error):
        max_prio = max(self.priorities.max(), td_error + 1e-6)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err.item()) + 1e-6


# --- Utility Functions ---
# Preprocess board: log2 + flatten + normalize
def preprocess(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    result = np.log2(state + 1).flatten() / 16.0
    return result

def find_high_tile(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    return np.max(state)

def compute_monotonicity(grid):
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)
    bonus = 0
    for row in grid:
        if all(x <= y for x, y in zip(row, row[1:])):
            bonus += 1
        if all(x >= y for x, y in zip(row, row[1:])):
            bonus += 1
    for col in zip(*grid):
        if all(x <= y for x, y in zip(col, col[1:])):
            bonus += 1
        if all(x >= y for x, y in zip(col, col[1:])):
            bonus += 1
    return bonus * 0.1

if __name__ == "__main__":

    # --- Training Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

    main_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters(), lr=1e-4)
    buffer = PrioritizedReplayBuffer(10000)

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
        total_reward = 0
        high_tile = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_tensor = torch.tensor([preprocess(state)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    action = main_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            bonus = compute_monotonicity(next_state)
            shaped_reward = reward + bonus
            done = terminated or truncated

            td_error = abs(shaped_reward)
            buffer.add((state, action, shaped_reward, next_state, done), td_error)
            state = next_state
            total_reward += reward
            high_tile = max(high_tile, np.max(state))

            if len(buffer.buffer) >= batch_size:
                samples, weights, indices = buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*samples)

                batch_states = torch.tensor([preprocess(s) for s in batch_states], dtype=torch.float32).to(device)
                batch_actions = torch.tensor(batch_actions).to(device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_states = torch.tensor([preprocess(s) for s in batch_next_states], dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)
                weights = weights.to(device)

                q_values = main_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                next_actions = main_net(batch_next_states).argmax(1, keepdim=True)
                next_q_values = target_net(batch_next_states).gather(1, next_actions).squeeze(1)
                target_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)

                loss = ((q_values - target_q_values.detach()) ** 2 * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                td_errors = torch.abs(q_values - target_q_values.detach())
                buffer.update_priorities(indices, td_errors)

            total_steps += 1
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode:03d} | Score: {total_reward:.0f} | High tile: {find_high_tile(state)} | Epsilon: {epsilon:.3f}")

    env.close()
